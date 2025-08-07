import openai
import pandas as pd
import os
import streamlit as st

try:
    api_key = st.secrets["TOGETHER_API_KEY"]
except:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("TOGETHER_API_KEY")




def makeDataFrame (sorted_indices, resume_texts, uploaded_files, top_n):
        
        topElements = []
        for i, score in sorted_indices[:top_n]:
            text = resume_texts[i]
            
            if text:
                words = text.strip().split()
                if len(words) >= 2:
                    candidate_name = words[0] + " " + words[1]
                elif len(words) == 1:
                    candidate_name = words[0]
                else:
                    candidate_name = "Unknown"
            else:
                candidate_name = "Unknown"

            topElements.append({
                "Candidate Name": candidate_name,
                "Similarity Score": f"{score:.2f}",
                "File Name": uploaded_files[i].name
            })


        # Return DataFrame
        return pd.DataFrame(topElements)




def ask_together_ai(prompt):
    client = openai.OpenAI(api_key=api_key, base_url="https://api.together.xyz/v1")

    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error: {e}"








def generatePrompt (sorted_indices, resume_texts, jd_text):
    best_index = sorted_indices[0][0]
    best_resume_text = resume_texts[best_index]

    # Prompt for AI
    return f"""Tell me why this person is a great fit for the role. Also let me know where the areas are that this person is lacking (separate paragraphs with headings). Your answer should be natural, and not more than 250 words. Use points if necessary and give a professional response.

---------------------------
Job Description:
{jd_text}

---------------------------
Resume:
{best_resume_text}
"""