import os
import streamlit as st
from langchain.llms import GooglePalm
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory

os.environ['GOOGLE_API_KEY'] = st.secrets['apikey']

# APP FRAMEWORK
st.title("🏕️Travel Advisor✈️")
prompt = st.text_input("AHOY!! WHERE IS YOUR NEXT ADVENTURE GONNA BE?")
place = st.text_input("Where are you staying or traveling from?")
button = st.button('Plan My Trip')

# DEFINE PROMPT TEMPLATES
title_template = PromptTemplate(
    input_variables = ['topic'],
    template = 'give me just the names of the best 5 places to visit in {topic} according to Lonely Planet?'
)

name_template = PromptTemplate(
    input_variables = ['topic'],
    template = 'Tell me the language spoken, the weather conditions, the temperature there and also the details about the 5 best places to visit in {topic} according to Lonely Planet?'
)

hotel_template = PromptTemplate(
    input_variables = ['title','place'],
    template = 'Give me the name and address of hotels with their prices per night(in the currency of {place}) and ratings near these places according to TripAdvisor and their links and arrange it in a table  : {title}'
)

flight_template = PromptTemplate(
    input_variables = ['title','place'],
    template = 'What are the cheapest flights to {title} from {place} and the prices of those flights(in the currency of {place}) with the stops they make and where these stops are and the total time of flight and also show the departing airport and the nearest arrival airport and arrange it in a table'
)

# MEMORY 

memory  = ConversationBufferMemory(input_key = 'topic', memory_key = 'chat_history')

# CONNECT TO LLM
llm = GooglePalm(temperature = 0.9)

# CREATE LLM CHAINS
title_chain = LLMChain(llm = llm, prompt = title_template, verbose = True, output_key = 'title', memory = memory)
name_chain = LLMChain(llm = llm, prompt = name_template, verbose = True, output_key = 'names', memory = memory)
hotel_chain = LLMChain(llm = llm, prompt = hotel_template, verbose = True, output_key = 'hotels', memory = memory)
flight_chain = LLMChain(llm = llm, prompt = flight_template, verbose = True, output_key = 'flights', memory = memory)

# CREATE SEQUENTIAL CHAIN
sequential_chain = SequentialChain(chains = [title_chain,name_chain,hotel_chain,flight_chain], input_variables = ['topic','place'], output_variables = ['title','names','hotels','flights'], verbose = True)


if button:

    if prompt and place:

        # RUN SEQUENTIAL CHAIN
        response = sequential_chain({'topic' : prompt, 'place' : place})

        # PRINT RESULTS
        st.write(response['names'])
        st.markdown("---")
        st.write(response['hotels'])
        st.markdown("---")
        st.write(response['flights'])
        st.markdown("---")

        #EXPANDER FOR DISPLAYING MEMORY
        with st.expander('Message History'):
            st.info(memory.buffer)

st.divider()

with st.expander('DISCLAIMER'):
    st.caption('THIS IS JUST A PROJECT MADE SOLELY FOR _EDUCATIONAL_ _PURPOSES_ !!')
    st.caption('This project utilizes the _PaLM_ _API_ , _Lonely_ _Planet_ and _Trip_ _Advisor_ to provide information')
    st.caption('Please note that the results generated by the API may be _inaccurate_ or _incomplete_') 
    st.caption('The information presented should not be considered as _absolute_ or _authoritative_, and it is advised to verify any critical details through reliable sources')
    st.caption('This was made for fun so test it out and enjoy')
    
st.divider()
