import pandas as pd
import numpy as np
from google.cloud import bigquery
import spacy
from collections import Counter
import re
from datetime import datetime, timedelta
import json
import os
import asyncio
from typing import Dict, List, Optional

class SupportTicketNLPProcessor:
    def __init__(self, project_id: str, dataset_id: str):
        self.project_id = project_id
        self.dataset_id = dataset_id
        
        # Initialize BigQuery client with credentials from environment
        credentials_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        if credentials_json:
            import json
            from google.oauth2 import service_account
            credentials_info = json.loads(credentials_json)
            credentials = service_account.Credentials.from_service_account_info(credentials_info)
            self.client = bigquery.Client(project=project_id, credentials=credentials)
        else:
            self.client = bigquery.Client(project=project_id)
        
        # Load spaCy model
        try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    print("✅ spaCy model loaded successfully!")
except (OSError, ImportError):
    print("⚠️ spaCy model not available - entity extraction disabled")
    nlp = None
        
        # Event-specific topic keywords
        self.topic_keywords = {
            'events': ['webinar', 'workshop', 'event', 'registration', 'attend', 'grafanacon', 'community call'],
            'kubernetes': ['kubernetes', 'k8s', 'container', 'pod', 'deployment', 'helm'],
            'prometheus': ['prometheus', 'promql', 'metrics', 'scraping', 'targets'],
            'dashboards': ['dashboard', 'panel', 'visualization', 'chart', 'graph', 'widget'],
            'alerts': ['alert', 'notification', 'alarm', 'warning', 'trigger', 'notify'],
            'observability': ['observability', 'monitoring', 'logging', 'tracing', 'telemetry'],
            'billing': ['billing', 'payment', 'invoice', 'charge', 'cost', 'price', 'subscription'],
            'authentication': ['login', 'password', 'auth', 'signin', 'sso', 'ldap', 'oauth'],
            'performance': ['slow', 'performance', 'timeout', 'lag', 'speed', 'loading', 'latency'],
            'integrations': ['integration', 'api', 'webhook', 'connector', 'plugin', 'export'],
            'data_sources': ['datasource', 'database', 'influxdb', 'elasticsearch', 'mysql', 'postgres']
        }
        
        # Intent and growth signal patterns
        self.intent_patterns = {
            'frustration': [
                r'\b(frustrated?|annoying|terrible|awful|hate|broken|useless)\b',
                r'\b(not working|doesn\'t work|won\'t work|can\'t|unable)\b',
                r'\b(urgent|asap|immediately|critical|emergency)\b'
            ],
            'curiosity': [
                r'\b(how to|how do|can you|is it possible|wondering|curious)\b',
                r'\b(learn|understand|tutorial|guide|documentation)\b',
                r'\b(what is|what does|explain|help me understand)\b'
            ],
            'expansion_interest': [
                r'\b(enterprise|team|multiple users|upgrade|premium|advanced)\b',
                r'\b(more features|additional|expand|scale|grow)\b',
                r'\b(custom|professional|business plan)\b'
            ],
            'technical_issue': [
                r'\b(error|bug|issue|problem|broken|failing|crash)\b',
                r'\b(500|404|timeout|connection|syntax|invalid)\b',
                r'\b(troubleshoot|debug|fix|resolve|solve)\b'
            ]
        }
        
        self.growth_signals = {
            'enterprise_interest': [
                r'\b(enterprise|sso|ldap|saml|compliance|audit)\b',
                r'\b(team management|user roles|permissions|admin)\b',
                r'\b(on-premise|private cloud|dedicated)\b'
            ],
            'scaling_up': [
                r'\b(more users|additional seats|team growth|expanding)\b',
                r'\b(higher limits|increase capacity|scale up)\b',
                r'\b(multiple environments|staging|production)\b'
            ],
            'integration_expansion': [
                r'\b(new integration|connect|sync|api access)\b',
                r'\b(custom metrics|custom dashboards|automation)\b',
                r'\b(webhook|export|reporting|analytics)\b'
            ]
        }

    def process_tickets_timeframe(self, days_back: int = 1, force_reprocess: bool = False) -> pd.DataFrame:
        """Process tickets from a specific timeframe"""
        
        # Build query based on parameters
        where_clause = f"WHERE created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days_back} DAY)"
        
        if not force_reprocess:
            # Only process tickets not already processed (or processed more than 24 hours ago)
            where_clause += f"""
            AND ticket_id NOT IN (
                SELECT ticket_id FROM `{self.project_id}.{self.dataset_id}.support_ticket_nlp` 
                WHERE processed_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
            )
            """
        
        query = f"""
        SELECT 
            ticket_id,
            user_id,
            message,
            topic,
            status,
            created_at,
            priority,
            first_response_time_hours,
            resolution_time_hours
        FROM `{self.project_id}.{self.dataset_id}.support_tickets`
        {where_clause}
        ORDER BY created_at DESC
        """
        
        try:
            tickets_df = self.client.query(query).to_dataframe()
        except Exception as e:
            print(f"Error querying BigQuery: {str(e)}")
            # Return empty DataFrame if query fails
            return pd.DataFrame()
        
        if len(tickets_df) == 0:
            print(f"No tickets to process for the last {days_back} days")
            return pd.DataFrame()
        
        print(f"Processing {len(tickets_df)} tickets from the last {days_back} days...")
        
        # Process each ticket
        results = []
        for _, ticket in tickets_df.iterrows():
            result = self._process_single_ticket(ticket)
            results.append(result)
        
        return pd.DataFrame(results)
    
    def _process_single_ticket(self, ticket) -> Dict:
        """Process a single ticket and return results"""
        message = ticket.get('message', '')
        
        # Extract insights using existing methods
        topics = self.extract_topics(message)
        intent = self.classify_intent(message)
        growth_signals = self.detect_growth_signals(message)
        entities = self.extract_entities(message) if self.nlp else {}
        urgency_score = self.calculate_urgency_score(message, ticket.get('priority', 'medium'))
        
        return {
            'ticket_id': ticket.get('ticket_id'),
            'user_id': ticket.get('user_id'),
            'original_topic': ticket.get('topic'),
            'detected_topics': topics,
            'primary_topic': topics[0] if topics else 'general',
            'intent': intent,
            'growth_signals': growth_signals,
            'urgency_score': round(urgency_score, 2),
            'entities': entities,
            'created_at': ticket.get('created_at'),
            'status': ticket.get('status'),
            'first_response_time_hours': ticket.get('first_response_time_hours'),
            'resolution_time_hours': ticket.get('resolution_time_hours'),
            'processed_at': datetime.now(),
            'has_growth_signal': len(growth_signals) > 0,
            'is_frustrated': intent == 'frustration',
            'is_expansion_interested': intent == 'expansion_interest'
        }
    
    def extract_topics(self, text):
        """Extract topics from ticket text using keyword matching"""
        if pd.isna(text):
            return ['general']
            
        text_lower = str(text).lower()
        detected_topics = []
        
        for topic, keywords in self.topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_topics.append(topic)
        
        return detected_topics if detected_topics else ['general']

    def classify_intent(self, text):
        """Classify the intent/sentiment of the ticket"""
        if pd.isna(text):
            return 'neutral'
            
        text_lower = str(text).lower()
        intents = []
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    intents.append(intent)
                    break
        
        # Return the most specific intent, or 'neutral' if none found
        if 'frustration' in intents:
            return 'frustration'
        elif 'expansion_interest' in intents:
            return 'expansion_interest'  
        elif 'technical_issue' in intents:
            return 'technical_issue'
        elif 'curiosity' in intents:
            return 'curiosity'
        else:
            return 'neutral'

    def detect_growth_signals(self, text):
        """Detect growth/expansion signals in tickets"""
        if pd.isna(text):
            return []
            
        text_lower = str(text).lower()
        signals = []
        
        for signal, patterns in self.growth_signals.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    signals.append(signal)
                    break
        
        return signals

    def extract_entities(self, text):
        """Extract named entities using spaCy"""
        if not self.nlp or pd.isna(text):
            return {}
            
        doc = self.nlp(str(text))
        entities = {
            'organizations': [ent.text for ent in doc.ents if ent.label_ == "ORG"],
            'products': [ent.text for ent in doc.ents if ent.label_ == "PRODUCT"],
            'technologies': [ent.text for ent in doc.ents if ent.label_ in ["PRODUCT", "ORG"]]
        }
        
        return entities

    def calculate_urgency_score(self, text, priority):
        """Calculate urgency score based on text and priority"""
        if pd.isna(text):
            text = ""
        
        urgency_words = [
            'urgent', 'asap', 'immediately', 'critical', 'emergency', 
            'broken', 'down', 'not working', 'production', 'outage'
        ]
        
        text_lower = str(text).lower()
        urgency_count = sum(1 for word in urgency_words if word in text_lower)
        
        # Base score from priority
        priority_scores = {'critical': 0.9, 'high': 0.7, 'medium': 0.4, 'low': 0.2}
        base_score = priority_scores.get(str(priority).lower(), 0.3)
        
        # Adjust based on urgency words
        urgency_adjustment = min(urgency_count * 0.1, 0.3)
        
        return min(base_score + urgency_adjustment, 1.0)
    
    def save_to_bigquery(self, results_df: pd.DataFrame, table_name: str = 'support_ticket_nlp'):
        """Save results to BigQuery with upsert logic"""
        if len(results_df) == 0:
            print("No results to save")
            return
        
        # Prepare data for BigQuery
        bq_df = results_df.copy()
        bq_df['detected_topics'] = bq_df['detected_topics'].apply(json.dumps)
        bq_df['growth_signals'] = bq_df['growth_signals'].apply(json.dumps)
        bq_df['entities'] = bq_df['entities'].apply(json.dumps)
        
        table_id = f"{self.project_id}.{self.dataset_id}.{table_name}"
        
        # Use WRITE_APPEND for incremental updates
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND",
            create_disposition="CREATE_IF_NEEDED",
            schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION]
        )
        
        print(f"Saving {len(bq_df)} processed tickets to {table_id}")
        try:
            job = self.client.load_table_from_dataframe(bq_df, table_id, job_config=job_config)
            job.result()
            print(f"✅ Successfully saved to BigQuery table: {table_id}")
        except Exception as e:
            print(f"❌ Error saving to BigQuery: {str(e)}")
            raise e
    
    def get_latest_insights(self) -> Dict:
        """Get latest processing insights from BigQuery"""
        query = f"""
        SELECT 
            COUNT(*) as total_tickets,
            COUNT(CASE WHEN has_growth_signal THEN 1 END) as growth_signals,
            COUNT(CASE WHEN is_frustrated THEN 1 END) as frustrated_customers,
            COUNT(CASE WHEN is_expansion_interested THEN 1 END) as expansion_interested,
            AVG(urgency_score) as avg_urgency_score,
            MAX(processed_at) as last_processed
        FROM `{self.project_id}.{self.dataset_id}.support_ticket_nlp`
        WHERE processed_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
        """
        
        try:
            result = self.client.query(query).to_dataframe()
            return result.iloc[0].to_dict() if len(result) > 0 else {}
        except Exception as e:
            print(f"Error getting insights: {str(e)}")
            return {}

    def generate_insights_summary(self, results_df: pd.DataFrame) -> Dict:
        """Generate summary insights from processed tickets"""
        if len(results_df) == 0:
            return {}
        
        total_tickets = len(results_df)
        
        # Topic distribution
        topic_dist = Counter()
        for topics in results_df['detected_topics']:
            topic_dist.update(topics)
        
        # Intent distribution
        intent_dist = results_df['intent'].value_counts()
        
        # Growth signals
        growth_signals_count = results_df['has_growth_signal'].sum()
        frustrated_users = results_df['is_frustrated'].sum()
        expansion_interested = results_df['is_expansion_interested'].sum()
        
        # Top urgency tickets
        high_urgency = results_df[results_df['urgency_score'] >= 0.7]
        
        summary = {
            'total_tickets_processed': total_tickets,
            'top_topics': dict(topic_dist.most_common(5)),
            'intent_distribution': intent_dist.to_dict(),
            'growth_signals_detected': int(growth_signals_count),
            'frustrated_customers': int(frustrated_users),
            'expansion_interested_customers': int(expansion_interested),
            'high_urgency_tickets': len(high_urgency),
            'average_urgency_score': round(results_df['urgency_score'].mean(), 2)
        }
        
        return summary
