import streamlit as st
from langchain_community.llms import Ollama
import re
import pandas as pd
from tabulate import tabulate
import numpy as np
import pickle
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns

# Function to extract Python code from Markdown text
def extract_problem(markdown_text):
    pattern = re.compile(r'```\n(.*?)\n```', re.DOTALL)
    matches = pattern.findall(markdown_text)
    return matches

# Function to save code blocks to a .py file
def save_problem(problem):

    with open('adaptivelearner/problem.md', 'w') as md_file:
        md_file.write(f'# Problem\n\n{problem}\n')

    # Define the code to append
    additional_code = """
        
########### ATTEMPT THE PROBLEM BELOW THIS ###############


########### ATTEMPT THE PROBLEM ABOVE THIS ###############

"""
    # Write the problem content and additional code to the file
    with open('adaptivelearner/tryproblem.py', 'w') as file:
        # file.write('"""\n')
        # file.write(f'{problem}\n')
        # file.write('"""\n')
        file.write(additional_code)
        
def extract_proficiency():
    df = pd.read_csv('adaptivelearner\proficiency.csv')
    markdown = tabulate(df, headers='keys', tablefmt='pipe', showindex=False)
    return markdown

# Prompt
def createproblem(subject, proficiency, specific_topic, llm):

    """
        input = proficiency
        output = problem
    """
    
    prompt = f"Goal: Your objective is to guide me through learning the subject of {subject} step by step on {specific_topic}. Utilize my current knowledge and proficiency in various topics, as outlined below, to craft an engaging example and a single coding quiz in the LeetCode format. Details: Proficiency Levels: {proficiency} Method: Create examples and corresponding coding problems tailored to my proficiency levels to facilitate incremental learning and mastery. Format: ONE example and ONE problem should be presented in the LeetCode style, with clear instructions and expected output.  Provide example, and A SINGLE problem that build on my current proficiency and challenge me in the {specific_topic}."
    out = llm.invoke(prompt)
    return out

# Before attempt prompt
def get_topic_options():
    # Read the proficiency file to get the topics
    df = pd.read_csv('adaptivelearner/proficiency.csv')
    
    # Ensure the DataFrame contains the required columns
    if 'topic' not in df.columns:
        raise ValueError("The CSV file does not contain the 'topic' column.")
    
    # Return unique topics
    return df['topic'].unique()

def topic_score_prompt(problem, topic):
    prompt = f"Assuming a scale of 0 to 10, how important is {topic} to understanding and solving {problem}? Provide a numerical score."
    out = llm.invoke(prompt)
    return out

def problem_topic_proficiency(problem, topic):
    out = topic_score_prompt(problem, topic)
    pattern = r'\b(\d)\b'
    match = re.search(pattern, out)
    if match:
        return int(match.group(1))
    else:
        return None

def update_problem_proficiency(specific_topic):
    
    df = pd.read_csv('adaptivelearner/proficiency.csv')
    df['proficiency'] = df.apply(lambda row: problem_topic_proficiency(row['topic'], problem) if row['topic'] == specific_topic else 0.0, axis=1)
    df.to_csv('adaptivelearner/problem_proficiency.csv', index=False)

# Check if problem is correct
def checkproblem():
    try:
        subprocess.check_call(['python', 'adaptivelearner/tryproblem.py'], stderr=subprocess.DEVNULL)
        output = 1
    except subprocess.CalledProcessError:
        output = -1

    #print(output)
    return output 

# After attempt prompt
def logit_normalize(x):
    # Normalize using the logit function between 0 and 1
    return 1 / (1 + np.exp(-x))

def update_proficiency(result, specific_topic):
    alpha = 0.1  # learning rate
    original_df = pd.read_csv('adaptivelearner\proficiency.csv')
    problem_csv_path = 'adaptivelearner\problem_proficiency.csv'
    problem_df = pd.read_csv(problem_csv_path)
    current_proficiency = original_df.loc[original_df['topic'] == specific_topic, 'proficiency'].values[0]
    problem_difficulty = problem_df.loc[problem_df['topic'] == specific_topic, 'proficiency'].values[0]
    difficulty_factor = problem_difficulty / 10  # normalize difficulty to [0, 1]
    update = alpha * difficulty_factor * result
    new_proficiency = (1-alpha)*current_proficiency + update
    new_proficiency = max(0.001, min(1, new_proficiency))  # clip to [0, 1]
    original_df.loc[original_df['topic'] == specific_topic, 'proficiency'] = new_proficiency
    original_df.to_csv('adaptivelearner\proficiency.csv', index=False)

def visualize_proficiency():
    # Read the CSV file into a DataFrame
    df = pd.read_csv('adaptivelearner\proficiency.csv')
    
    # Ensure the DataFrame contains the required columns
    if 'topic' not in df.columns or 'proficiency' not in df.columns:
        raise ValueError("The CSV file does not contain the required columns: 'topic' and 'proficiency'.")
    
    # Create a plot
    plt.figure(figsize=(10, 8))
    sns.barplot(data=df, x='topic', y='proficiency', palette='viridis')
    
    # Add titles and labels
    plt.title('Proficiency by Topic')
    plt.xlabel('Topic')
    plt.ylabel('Proficiency')
    plt.xticks(rotation=45, ha='right')  # Rotate topic labels if needed

    # Save the plot as an image
    plt.tight_layout()  # Adjust layout to fit everything
    plt.savefig('adaptivelearner\proficiency.png')
    plt.close()

# Streamlit app
st.title("Personal Adaptive Learner")
st.write("Developed by [Srijit Mukherjee](https://www.linkedin.com/in/srijit-mukherjee/). Code available at [Github](https://github.com/mukherjeesrijit/adaptivelearner).")

# Initialize the LLM
llm = Ollama(model="llama3.1")

# Get the subject and proficiency
subject = "Pytorch Coding for Deep Learning"
proficiency = extract_proficiency()

# Get the topic options
topic_options = get_topic_options()

##################################################################################

# Create tabs
tab1, tab2, tab3 = st.tabs(["Problem", "Solve", "Proficiency"])

# Create a dropdown for the topic selection in tab1
with tab1:
    st.write("Select a topic:")
    specific_topic = st.selectbox("Topic", topic_options)

    # Create the problem
    if st.button("Create Problem"):
        problem = createproblem(subject, proficiency, specific_topic, llm)
        save_problem(problem)
        update_problem_proficiency(specific_topic)
        with open('topic.pkl', 'wb') as f:
            pickle.dump(specific_topic, f)
        st.write("Problem created!")

    # Display the problem
    with open('adaptivelearner/problem.md', 'r') as f:
        problem_text = f.read()
    st.markdown(problem_text)

# Create a text box for the user to enter their code in tab2
with tab2:
    st.write("Enter your code:")
    code = st.text_area("Code", height=200)

    # Save the code to a file
    if st.button("Save Code"):
        with open('adaptivelearner/tryproblem.py', 'w') as f:
            f.write(code)
        st.write("Code saved!")

    # Run the code and check if it's correct
    if st.button("Run Code"):
        
        original_df = pd.read_csv('adaptivelearner\proficiency.csv')
        previous_proficiency = original_df.loc[original_df['topic'] == specific_topic, 'proficiency'].values[0]
        print(previous_proficiency)

        result = checkproblem()
        if result == 1:
            st.write("Correct!")
        else:
            st.write("Incorrect")

        # Update the proficiency score
        update_proficiency(result, specific_topic)
        original_df = pd.read_csv('adaptivelearner\proficiency.csv')
        new_proficiency = original_df.loc[original_df['topic'] == specific_topic, 'proficiency'].values[0]
        print(new_proficiency)
        # Display a pop-up with the change in proficiency
        delta = (100*(new_proficiency - previous_proficiency))/previous_proficiency
        if delta > 0:
            st.success(f"Proficiency increased by {delta*100:.2f}%!")
        elif delta < 0:
            st.error(f"Proficiency decreased by {abs(delta)*100:.2f}%. Try again!")
        else:
            st.info("Proficiency remains the same.")

# Display the updated proficiency score in tab3
with tab3:
    visualize_proficiency()
    st.image('adaptivelearner/proficiency.png')
