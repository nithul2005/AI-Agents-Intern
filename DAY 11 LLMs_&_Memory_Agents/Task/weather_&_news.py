import os
import requests
import gradio as gr
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY")
NEWS_API_KEY = os.environ.get("NEWS_API_KEY")

print("GROQ loaded:", bool(GROQ_API_KEY))

llm = ChatGroq(temperature=0.3, model_name="llama3-8b-8192")

news_prompt = PromptTemplate(
    input_variables=["news"],
    template="""
You are an intelligent assistant. Summarize the following news headlines in a concise and engaging manner for a general audience:

{news}
"""
)

# Prompt template for weather beautification
weather_prompt = PromptTemplate(
    input_variables=["city", "weather"],
    template="""
You are a friendly assistant. Based on the weather info provided below, give a natural language response about the current weather in {city}.

Weather data:
{weather}
"""
)

# Chains
weather_chain = LLMChain(llm=llm, prompt=weather_prompt)
news_chain = LLMChain(llm=llm, prompt=news_prompt)

def get_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    res = requests.get(url)
    if res.status_code != 200:
        return None
    data = res.json()
    return f"Temperature: {data['main']['temp']}°C, Condition: {data['weather'][0]['description'].capitalize()}"

def get_top_news():
    url = f"https://newsapi.org/v2/top-headlines?language=en&pageSize=5&apiKey={NEWS_API_KEY}"
    res = requests.get(url)
    if res.status_code != 200:
        return None
    articles = res.json().get("articles", [])
    return "\n".join([f"{i+1}. {a['title']}" for i, a in enumerate(articles)])

def assistant(city):
    weather_raw = get_weather(city)
    if not weather_raw:
        weather_result = "❌ Could not fetch weather. Check city name or API key."
    else:
        weather_result = weather_chain.run(city=city, weather=weather_raw)

    news_raw = get_top_news()
    if not news_raw:
        news_result = "❌ Could not fetch news. Check your NewsAPI key."
    else:
        news_result = news_chain.run(news=news_raw)

    return weather_result, news_result

iface = gr.Interface(
    fn=assistant,
    inputs=gr.Textbox(placeholder="Enter city name", label="City"),
    outputs=[
        gr.Textbox(label="🌤️ Live Weather Update"),
        gr.Textbox(label="📰 Global Trending News Summary")
    ],
    title="🌍 Live Weather + Global News Assistant",
    description="Enter a city name to get live weather and summarized top global news."
)

if __name__ == "__main__":
    iface.launch()
