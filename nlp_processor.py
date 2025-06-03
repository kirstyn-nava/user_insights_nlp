import pandas as pd
import numpy as np
import spacy
from collections import Counter
import re
from datetime import datetime
import json

# Load spaCy model for NLP processing
# Install with: python -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_sm")
    print("✅ spaCy model loaded successfully!")
except OSError:
    print("❌ Please install spaCy English model: python -m spacy download en_core_web_sm")
    nlp = None

class SupportTicketNLPParser:
    def __init__(self):
        # Define topic keywords for classification
        self.topic_keywords = {
            'alerts': ['alert', 'notification', 'alarm', 'warning', 'trigger', 'notify'],
            'dashboards': ['dashboard', 'panel', 'visualization', 'chart', 'graph', 'widget'],
            'billing': ['billing', 'payment', 'invoice', 'charge', 'cost', 'price', 'subscription'],
            'authentication': ['login', 'password', 'auth', 'signin', 'sso', 'ldap', 'oauth'],
            'performance': ['slow', 'performance', 'timeout', 'lag', 'speed', 'loading', 'latency'],
            'integrations': ['integration', 'api', 'webhook', 'connector', 'plugin', 'export'],
            'data_sources': ['datasource', 'database', 'prometheus', 'influxdb', 'elasticsearch'],
            'scaling': ['scale', 'enterprise', 'team', 'users', 'capacity', 'limits', 'upgrade']
        }
        
        # Intent classification patterns
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
        
        # Growth signal patterns
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

    def load_support_tickets(self, csv_path='support_tickets.csv'):
        """Load support tickets from CSV file"""
        try:
            df = pd.read_csv(csv_path)
            print(f"✅ Loaded {len(df)} tickets from {csv_path}")
            return df
        except FileNotFoundError:
            print(f"❌ Could not find {csv_path}")
            print("Make sure the support_tickets.csv file is in the same directory as this script")
            return None

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
        if not nlp or pd.isna(text):
            return {}
            
        doc = nlp(str(text))
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

    def process_tickets(self, csv_path='support_tickets.csv'):
        """Main processing function"""
        print("🤖 Starting NLP processing of support tickets...")
        
        # Load tickets from CSV
        tickets_df = self.load_support_tickets(csv_path)
        if tickets_df is None:
            return None
        
        print(f"Processing {len(tickets_df)} tickets...")
        
        # Process each ticket
        results = []
        for idx, ticket in tickets_df.iterrows():
            message = ticket.get('message', '')
            
            # Extract insights
            topics = self.extract_topics(message)
            intent = self.classify_intent(message)
            growth_signals = self.detect_growth_signals(message)
            entities = self.extract_entities(message)
            urgency_score = self.calculate_urgency_score(message, ticket.get('priority', 'medium'))
            
            result = {
                'ticket_id': ticket.get('ticket_id', f'ticket_{idx}'),
                'user_id': ticket.get('user_id', ''),
                'original_topic': ticket.get('topic', ''),
                'detected_topics': topics,
                'primary_topic': topics[0] if topics else 'general',
                'intent': intent,
                'growth_signals': growth_signals,
                'urgency_score': round(urgency_score, 2),
                'entities': entities,
                'created_at': ticket.get('created_at', datetime.now()),
                'status': ticket.get('status', ''),
                'processed_at': datetime.now(),
                'original_message': message
            }
            
            results.append(result)
            
            # Show progress every 100 tickets
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(tickets_df)} tickets...")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Add some aggregated insights
        results_df['has_growth_signal'] = results_df['growth_signals'].apply(lambda x: len(x) > 0)
        results_df['is_frustrated'] = results_df['intent'] == 'frustration'
        results_df['is_expansion_interested'] = results_df['intent'] == 'expansion_interest'
        
        print("✅ NLP processing complete!")
        return results_df

    def save_to_csv(self, results_df, output_path='support_ticket_nlp_results.csv'):
        """Save results to CSV"""
        # Convert lists to JSON strings for CSV
        csv_df = results_df.copy()
        csv_df['detected_topics'] = csv_df['detected_topics'].apply(json.dumps)
        csv_df['growth_signals'] = csv_df['growth_signals'].apply(json.dumps)
        csv_df['entities'] = csv_df['entities'].apply(json.dumps)
        
        csv_df.to_csv(output_path, index=False)
        print(f"💾 Results saved to {output_path}")

    def generate_insights_summary(self, results_df):
        """Generate summary insights from processed tickets"""
        
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

    def show_sample_results(self, results_df, n=5):
        """Display sample results"""
        print(f"\n📋 Sample of {n} processed tickets:")
        print("="*80)
        
        sample = results_df.head(n)
        for _, ticket in sample.iterrows():
            print(f"Ticket ID: {ticket['ticket_id']}")
            print(f"Original Message: {ticket['original_message'][:100]}...")
            print(f"Primary Topic: {ticket['primary_topic']}")
            print(f"Intent: {ticket['intent']}")
            print(f"Urgency Score: {ticket['urgency_score']}")
            print(f"Growth Signals: {ticket['growth_signals']}")
            print("-"*40)

# Example usage
if __name__ == "__main__":
    # Initialize the parser
    parser = SupportTicketNLPParser()
    
    # Process tickets from CSV
    results = parser.process_tickets('support_tickets.csv')
    
    if results is not None:
        # Save results
        parser.save_to_csv(results)
        
        # Generate insights
        insights = parser.generate_insights_summary(results)
        
        print("\n🎯 === NLP PROCESSING COMPLETE ===")
        print(f"📊 Summary Insights:")
        for key, value in insights.items():
            print(f"  {key}: {value}")
        
        # Show sample results
        parser.show_sample_results(results)
        
        print(f"\n✅ All done! Check 'support_ticket_nlp_results.csv' for full results.")
    else:
        print("❌ Processing failed. Please check your CSV file.")
