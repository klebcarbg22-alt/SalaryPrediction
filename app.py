import streamlit as st
import time
import os
from dotenv import load_dotenv
from openai import OpenAI

from safety import is_safe
from evaluation import evaluate_response
from logger import log_interaction


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Responsible AI Chatbot",
    page_icon="🤖",
    layout="wide"
)


# -----------------------------
# Custom CSS UI
# -----------------------------
st.markdown("""
<style>

/* Background */
[data-testid="stAppViewContainer"] {
    background: #f8fafc;
}


/* Header Card */
.header {

    background: linear-gradient(
        135deg,
        #2563eb,
        #06b6d4
    );

    padding: 30px;

    border-radius: 20px;

    text-align: center;

    color:white;

    box-shadow:0px 8px 20px rgba(0,0,0,0.15);

    margin-bottom:30px;
}


.header h1 {

    color:white !important;

    font-size:42px;

    margin:0;

}


.header p {

    font-size:18px;

}


/* Input Box */

.stTextInput input {

    border-radius:15px;

    border:2px solid #2563eb;

    padding:12px;

    font-size:16px;

}


/* Generate Button */

.stButton button {

    background:linear-gradient(
        90deg,
        #2563eb,
        #06b6d4
    );

    color:white;

    border:none;

    border-radius:15px;

    height:45px;

    width:100%;

    font-size:18px;

    font-weight:bold;

}


.stButton button:hover {

    background:#1d4ed8;

}



/* History Title */

h2 {

    color:#1e3a8a !important;

}



/* Chat Card */

.chat-card {

    background:white;

    padding:20px;

    border-radius:20px;

    margin:15px 0;

    box-shadow:
    0 5px 15px rgba(0,0,0,0.1);

}


/* User */

.user {

    color:#16a34a;

    font-size:18px;

    font-weight:bold;

}


/* AI */

.ai {

    color:#2563eb;

    font-size:18px;

    font-weight:bold;

}


/* Status */

.status {

    color:#64748b;

    font-size:14px;

}



/* Sidebar */

section[data-testid="stSidebar"] {

    background:#0f172a;

}


section[data-testid="stSidebar"] * {

    color:white !important;

}



</style>
""",
unsafe_allow_html=True)



# -----------------------------
# Load API Key
# -----------------------------
load_dotenv()


client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)


MODEL = "openai/gpt-4o-mini"



# -----------------------------
# Query Model
# -----------------------------
def query_model(prompt):

    response = client.chat.completions.create(

        model=MODEL,

        messages=[

            {
                "role":"system",
                "content":
                "You are a helpful, safe, and responsible AI assistant."
            },

            {
                "role":"user",
                "content":prompt
            }

        ],

        temperature=0.7,

        max_tokens=300

    )


    return response.choices[0].message.content




# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="header">

<h1>🤖 Responsible AI Chatbot</h1>

<p>
✅ Safe &nbsp; | &nbsp;
✅ Evaluated &nbsp; | &nbsp;
✅ Logged
</p>

</div>
""",
unsafe_allow_html=True)



# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:

    st.title("⚙️ AI Settings")

    st.info(
        """
        This AI assistant provides:
        
        ✔ Safe responses
        
        ✔ Response evaluation
        
        ✔ Interaction logging
        """
    )


    st.write("Model")

    st.code(MODEL)



# -----------------------------
# Memory
# -----------------------------
if "history" not in st.session_state:

    st.session_state.history = []



# -----------------------------
# User Input
# -----------------------------
user_input = st.text_input(
    "💬 Ask something:",
    placeholder="Ask about AI, coding, science..."
)



if st.button("🚀 Generate Response"):


    if not user_input.strip():

        st.warning(
            "Please enter a question."
        )



    elif not is_safe(user_input):


        response = (
            "I'm sorry that you're going through something difficult. "
            "If you're feeling overwhelmed or thinking about harming yourself, "
            "please consider talking with someone you trust."
        )


        quality = "⚠️ Unsafe Prompt"

        elapsed = 0.0



        log_interaction(
            user_input,
            response,
            quality,
            elapsed
        )



        st.session_state.history.append(
            (
                user_input,
                response,
                quality,
                elapsed
            )
        )



    else:


        start = time.time()


        with st.spinner("🤖 AI is thinking..."):

            try:

                response = query_model(user_input)


            except Exception as e:

                response = f"❌ API Error: {e}"



        elapsed = time.time() - start


        quality = evaluate_response(response)



        log_interaction(
            user_input,
            response,
            quality,
            elapsed
        )



        st.session_state.history.append(
            (
                user_input,
                response,
                quality,
                elapsed
            )
        )




# -----------------------------
# Chat History
# -----------------------------
st.subheader("💬 Conversation History")


if not st.session_state.history:

    st.info(
        "No conversation yet. Start chatting with your AI assistant!"
    )



for user, ai, quality, t in reversed(
    st.session_state.history
):


    st.markdown(
        f"""

<div class="chat-card">


<div class="user">
🧑 You
</div>

<p>{user}</p>



<div class="ai">
🤖 AI Assistant
</div>

<p>{ai}</p>



<div class="status">
{quality} | ⏱ {t:.2f}s
</div>


</div>

""",
unsafe_allow_html=True
)