import streamlit as st
import pandas as pd
import json
from nlp_processor import SupportTicketNLPParser
import io

st.set_page_config(page_title="Support Signal Miner", page_icon="🎯", layout="wide")

st.title("🎯 Support Signal Miner")
st.subheader("Real-time NLP processing of support tickets")

# Initialize parser
@st.cache_resource
def load_parser():
    return SupportTicketNLPParser()

parser = load_parser()

# Realistic demo tickets
sample_tickets = {
    "Hi, we're seeing some weird behavior with our main dashboard since yesterday. The CPU metrics are showing but memory graphs are completely blank, and our alerts aren't firing even though we know memory usage is high. This is our primary production monitoring dashboard that the whole engineering team relies on. Can someone help us figure out what's going on? We've tried refreshing but no luck.",
    
    "Hey team! We just closed our Series B and are planning to grow our engineering org from 15 to about 40 people over the next 6 months. Right now we have everyone on the free plan but we're starting to hit some limits with datasources and team management. Could someone walk me through what the team/enterprise options look like? Specifically interested in user management and permissions since we'll have multiple teams working on different projects.",
    
   "I'm trying to set up our Kubernetes cluster monitoring and I've got Prometheus running, but I'm having trouble getting the connection working in Grafana. The docs mention a few different approaches and I'm not sure which one makes the most sense for our setup. We're running on AWS EKS if that matters. Has anyone written up best practices for this kind of setup? Thanks!",
    
   "Quick question - we're using Datadog for APM but want to bring our infrastructure metrics into Grafana for a unified view. I see there's a Datadog integration but wondering about any gotchas before we dive in. Also curious if there's a way to sync our existing Datadog alerts or if we need to rebuild them? Our DevOps team is already pretty stretched so trying to minimize the migration overhead. Appreciate any guidance!"
}

# Sidebar for navigation
st.sidebar.title("Demo Options")
demo_mode = st.sidebar.radio("Choose demo mode:", 
    ["Single Ticket Analysis", "Batch Processing"])

if demo_mode == "Single Ticket Analysis":
    st.header("📝 Single Ticket Analysis")
    
    # Quick demo buttons
    st.write("**Quick Examples:**")
    cols = st.columns(2)
    with cols[0]:
        for i, (label, text) in enumerate(list(sample_tickets.items())[:2]):
            if st.button(label, key=f"btn_{i}"):
                st.session_state.ticket_text = text

    with cols[1]:
        for i, (label, text) in enumerate(list(sample_tickets.items())[2:], 2):
            if st.button(label, key=f"btn_{i}"):
                st.session_state.ticket_text = text

    # Text input
    ticket_text = st.text_area(
        "Or paste your own support ticket:", 
        value=st.session_state.get('ticket_text', ''), 
        height=120,
        placeholder="Paste a support ticket here to see NLP analysis..."
    )

    if ticket_text:
        with st.spinner("🤖 Processing with SpaCy NLP..."):
            # Process with your existing methods
            topics = parser.extract_topics(ticket_text)
            intent = parser.classify_intent(ticket_text)
            growth_signals = parser.detect_growth_signals(ticket_text)
            entities = parser.extract_entities(ticket_text)
            urgency = parser.calculate_urgency_score(ticket_text, 'medium')
            
            # Results display
            st.header("🎯 Analysis Results")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Primary Topic", topics[0] if topics else "general")
            with col2:
                st.metric("Intent", intent.replace('_', ' ').title())
            with col3:
                st.metric("Urgency Score", f"{urgency:.2f}")
            with col4:
                st.metric("Growth Signals", len(growth_signals))
            
            # Detailed breakdown
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**📊 All Topics Detected:**")
                st.write(", ".join(topics) if topics else "general")
                
                if growth_signals:
                    st.write("**📈 Growth Signals:**")
                    for signal in growth_signals:
                        st.write(f"• {signal.replace('_', ' ').title()}")
            
            with col2:
                if entities.get('organizations') or entities.get('technologies'):
                    st.write("**🏢 Entities Found:**")
                    if entities.get('organizations'):
                        st.write(f"Organizations: {', '.join(entities['organizations'])}")
                    if entities.get('technologies'):
                        st.write(f"Technologies: {', '.join(entities['technologies'])}")
                
                # Segment recommendation
                st.write("**🎯 Recommended Segment:**")
                if intent == 'expansion_interest' or growth_signals:
                    st.success("→ High-Value Expansion Prospects")
                elif intent == 'frustration' and urgency > 0.7:
                    st.error("→ At-Risk Customer - Priority Support")
                elif intent == 'curiosity':
                    st.info("→ Education-Ready Users")
                else:
                    st.write("→ General Support Queue")

else:  # Batch Processing
    st.header("📊 Batch Processing Demo")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file with support tickets", type="csv")
    
    if uploaded_file:
        # Read CSV
        df = pd.read_csv(uploaded_file)
        st.write(f"📁 Loaded {len(df)} tickets")
        st.write("**Preview:**")
        st.dataframe(df.head())
        
        if st.button("🚀 Process All Tickets"):
            with st.spinner(f"Processing {len(df)} tickets..."):
                # Save uploaded file temporarily
                df.to_csv('temp_tickets.csv', index=False)
                
                # Process with your existing method
                results = parser.process_tickets('temp_tickets.csv')
                
                if results is not None:
                    # Generate insights
                    insights = parser.generate_insights_summary(results)
                    
                    # Display results
                    st.success("✅ Processing Complete!")
                    
                    # Key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Processed", insights['total_tickets_processed'])
                    with col2:
                        st.metric("Growth Signals", insights['growth_signals_detected'])
                    with col3:
                        st.metric("Frustrated Users", insights['frustrated_customers'])
                    with col4:
                        st.metric("Avg Urgency", insights['average_urgency_score'])
                    
                    # Charts
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**📊 Top Topics:**")
                        topic_df = pd.DataFrame(list(insights['top_topics'].items()), 
                                              columns=['Topic', 'Count'])
                        st.bar_chart(topic_df.set_index('Topic'))
                    
                    with col2:
                        st.write("**😊 Intent Distribution:**")
                        intent_df = pd.DataFrame(list(insights['intent_distribution'].items()), 
                                               columns=['Intent', 'Count'])
                        st.bar_chart(intent_df.set_index('Intent'))
                    
                    # Download results
                    csv_buffer = io.StringIO()
                    # Convert results for CSV download
                    csv_results = results.copy()
                    csv_results['detected_topics'] = csv_results['detected_topics'].apply(json.dumps)
                    csv_results['growth_signals'] = csv_results['growth_signals'].apply(json.dumps)
                    csv_results['entities'] = csv_results['entities'].apply(json.dumps)
                    csv_results.to_csv(csv_buffer, index=False)
                    
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv_buffer.getvalue(),
                        file_name="support_ticket_nlp_results.csv",
                        mime="text/csv"
                    )

with st.expander("🚀 How This Powers Dynamic Segmentation"):
    st.write("""
    **This NLP analysis automatically creates intelligent user segments:**
    
    • **High-Value Expansion Prospects** → Enterprise event invitations
    • **At-Risk Customers** → Reduced email frequency + priority support  
    • **Education-Ready Users** → Tutorial webinars and documentation
    • **Technical Power Users** → Advanced feature announcements
    
    All segments update in real-time as new support tickets are processed, 
    feeding directly into Customer.io for personalized campaign targeting.
    """)

# Footer
st.markdown("---")
st.markdown("*Built with SpaCy NLP • Powered by DigitalOcean App Platform*")
