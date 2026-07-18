from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
load_dotenv()


api_key = os.getenv("GOOGLE_API_KEY")

google_model = ChatGoogleGenerativeAI(
				model="gemini-3.1-flash-lite",
				api_key=api_key,
				temperature=1
	)