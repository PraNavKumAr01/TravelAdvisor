import os
import streamlit as st
from langchain.llms import GooglePalm
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory

GOOGLE_API_KEY = apikey

# APP FRAMEWORK
st.title("🏕️Travel Advisor✈️")
prompt = st.text_input("AHOY!! WHERE IS YOUR NEXT ADVENTURE GONNA BE?")
place = st.text_input("Where are you staying or traveling from?")

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
