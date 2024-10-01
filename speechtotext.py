import os, time, sys, io
import whisper
from torch import *
from speakerNameStruct import SpeakerNameWithText
from readFile import WriteFile

# from newsapi import NewsApiClient
# import ast
import json
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import Pinecone, ServerlessSpec
# from transformers import BertTokenizer, BertModel, AutoModel
from pinecone_text.sparse import BM25Encoder
import nltk
nltk.download('punkt_tab')
from openai import OpenAI
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, \
                         DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from sentence_transformers import SentenceTransformer
import wikipedia

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

#pinecone stuff 
pc = Pinecone(api_key="553e01e3-7e1a-46f3-ba75-098d1d9288df")
index1 = pc.Index("factcheck-article-data-tvisha-avhs")
index2 = pc.Index("vp-factcheck-article-data-tvisha-avhs")

client = OpenAI(
    api_key = "{{ secrets.OPENAI_API_KEY }}"
)

bm25 = BM25Encoder()
model = SentenceTransformer('all-mpnet-base-v2')


def generate_embeddings(texts : list[str]):
    fitted = bm25.fit(texts)
    embeddings= bm25.encode_documents(texts)

    return embeddings

def generate_dense(sentences):
    embeddings = model.encode(sentences)
    return embeddings

def get_context_wiki(statement):
    context = ""
    try:
        result = wikipedia.search(statement, results = 3)

        for r in result:
            context += wikipedia.summary(r)

        return context
    except:
        return "no information"


def get_context_news(topic):
    result1 = index1.query(
        vector = generate_dense(topic).tolist(),
        top_k=10,
        include_values=False,
        include_metadata=True
    )
    result2 = index2.query(
        vector = generate_dense(topic).tolist(),
        top_k=10,
        include_values=False,
        include_metadata=True
    )

    result = result1 + result2

    matched_info = ' '.join(item['metadata']['text'] for item in result['matches'])

    context = f"Information: {matched_info}"

    return context


def get_sufficiency(previous_statements_and_answers, new_statement, context):

    INSTRUCTIONS = "You are going to receive a statement from a political speech formatted \"speaker: STATEMENT\". Look at the information from the following LATEST NEWS ARTICLES: " + context + " and add it to your existing knowledge base. Then, ONLY IF THE STATEMENT, AS SPOKEN BY THE MENTIONED SPEAKER IS SIGNIFICANT AND NOT AN OPINION, determine if the information, along with any previous knowledge you have on the topic, and your previous chat history, is enough to even SLIGHTLY IMPLY whether ANY PART OF the STATEMENT is EITHER true or false, AS SPOKEN BY THE MENTIONED SPEAKER. DO NOT TRY TO DETERMINE IF THE STATEMENT IS TRUE OR FALSE, ONLY WHETHER THERE IS ENOUGH INFORMATION. TAKE AN OPTIMISTIC STANDPOINT. THERE IS MOST LIKELY ENOUGH INFORMATION. IT DOES NOT NEED TO BE TOO SPECIFIC. DO NOT TRY TO DETERMINE IF THE STATEMENT IS TRUE OR FALSE, ONLY RESPOND WITH WHETHER THERE IS ENOUGH INFORMATION. OUTPUT A JSON RESPONSE with the STATEMENT, whether there is enough information: True if there is, False if there isn't, insignificant IF THE STATEMENT IS INSIGNIFICANT, or opinion IF THE STATEMENT IS STATED AS AN OPINION (IF A STATEMENT IS AN OPINION BUT NOT STATED IN THE FORM OF AN OPINION, DO NOT CLASSIFY IT AS OPINION), AND THE REASON FOR YOUR DETERMINATION. KEEP IN MIND DO NOT TRY TO DETERMINE IF THE STATEMENT IS TRUE OR FALSE. ONLY RESPOND WITH WHETHER THERE IS ENOUGH INFORMATION. EVEN IF THE PROVIDED INFORMATION SUGGESTS THAT THE STATEMENT IS FALSE, SINCE YOU ARE ABLE TO MAKE THAT CONCLUSION, THAT MEANS THERE IS ENOUGH INFORMATION SO YOU SHOULD RETURN TRUE. Ensure the JSON output uses DOUBLE QUOTES for keys and string values ONLY. Ensure the JSON output uses SINGLE QUOTES for anything else. Formatting example: {\"statement\": \"STATEMENT\", \"sufficient_information\": \"True\"/\"False\"/\"opinion\"/\"insignificant\", \"reason\": \"reasoning\"}. Return responses in this format and this format only. Include nothing else in your response."
    TEMPERATURE = 0.5
    FREQUENCY_PENALTY = 0
    PRESENCE_PENALTY = 0.6
    MAX_CONTEXT_QUESTIONS = 10

    # build the messages
    messages = [
        { "role": "system", "content": INSTRUCTIONS },
    ]
    # add the previous questions and answers
    for statement, answer in previous_statements_and_answers[-MAX_CONTEXT_QUESTIONS:]:
        messages.append({ "role": "user", "content": statement })
        messages.append({ "role": "assistant", "content": answer })
    # add the new question
    messages.append({ "role": "user", "content": new_statement })

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=TEMPERATURE,
        top_p=1,
        frequency_penalty=FREQUENCY_PENALTY,
        presence_penalty=PRESENCE_PENALTY,
    )
    # print(completion.choices[0].message.content)
    return completion.choices[0].message.content

def get_topic(previous_statements_and_answers, new_statement):

    INSTRUCTIONS = "You are going to receive a statement from a political speech formatted 'speaker: STATEMENT'. Return the most prevalent topic in the STATEMENT as a Python string. Only return the topic and nothing else"
    TEMPERATURE = 0.5
    FREQUENCY_PENALTY = 0
    PRESENCE_PENALTY = 0.6
    MAX_CONTEXT_QUESTIONS = 10

    # build the messages
    messages = [
        { "role": "system", "content": INSTRUCTIONS },
    ]
    # add the previous questions and answers
    for statement, answer in previous_statements_and_answers[-MAX_CONTEXT_QUESTIONS:]:
        messages.append({ "role": "user", "content": statement })
        messages.append({ "role": "assistant", "content": answer })
    # add the new question
    messages.append({ "role": "user", "content": new_statement })

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=TEMPERATURE,
        top_p=1,
        frequency_penalty=FREQUENCY_PENALTY,
        presence_penalty=PRESENCE_PENALTY,
    )

    return completion.choices[0].message.content

def factcheck(previous_statements_and_answers, new_statement, context):

    INSTRUCTIONS = "Context: " + context + "Instructions: You are a factchecker. You are going to receive a statement from a political speech formatted 'speaker: STATEMENT'. When you get a message, PRIORITIZE tHE CONTEXT PROVIDED for accurate and specific information but also use your preexisting knowledge along with the chat history to output a json response with the STATEMENT, whether the STATEMENT is true or false, and the reason for your determination. Take into account whether something could be true. If it is probable that this statement could true even though there is no explicit record or evidence of it, take that into account instead of outrightly declaring it false. formatting example: {'speaker: speaker, 'statement': STATEMENT, 'result': False, 'reason': 'because the statement is false'} Return responses in this format and this format only. Include nothing else in your response."
    TEMPERATURE = 0.5
    FREQUENCY_PENALTY = 0
    PRESENCE_PENALTY = 0.6
    MAX_CONTEXT_QUESTIONS = 10

    # build the messages
    messages = [
        { "role": "system", "content": INSTRUCTIONS },
    ]
    # add the previous questions and answers
    for statement, answer in previous_statements_and_answers[-MAX_CONTEXT_QUESTIONS:]:
        messages.append({ "role": "user", "content": statement })
        messages.append({ "role": "assistant", "content": answer })
    # add the new question
    messages.append({ "role": "user", "content": new_statement })

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=TEMPERATURE,
        top_p=1,
        frequency_penalty=FREQUENCY_PENALTY,
        presence_penalty=PRESENCE_PENALTY,
    )

    # print(completion.choices[0].message.content)
    return completion.choices[0].message.content


# def remove_apostrophes_from_reason(json_str):
#     # Ensure the JSON string uses double quotes for valid JSON
#     json_str = json_str.replace("'", '"')

#     # Regex pattern to match the 'reason' field and capture its content
#     pattern = r'("reason":\s*"([^"]*)")'
    
#     def remove_apostrophes(match):
#         # Extract the reasoning part
#         reason_content = match.group(2)
#         # Remove or escape inner apostrophes in the reason content
#         modified_reason = reason_content.replace("\"", "")
#         # Reconstruct the 'reason' field with the modified content
#         return f'"reason": "{modified_reason}"'
    
#     # Replace the 'reason' field in the JSON string with the modified version
#     fixed_json_str = re.sub(pattern, remove_apostrophes, json_str)
    
#     return fixed_json_str


# Load the model (options: 'tiny', 'base', 'small', 'medium', 'large')
w_model = whisper.load_model("medium")

isRunning = True

# file to store transcription / audio path -> USE '/' AND NOT '\'
total_transcript = ""
transcript_list = []
temp_transcript = ""
transcript_path = "transcript.txt"
home_dir = os.path.expanduser("~")
relative_path = "\\OneDrive\\Desktop\\NAudio\\"
fact_checks_path = "fact_checks.txt"

history_sufficiency = []
history_topic = []
history_factcheck = []

## Speaker Diarization Model
# pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.0", use_auth_token="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")

#   **TO USE** 
#   
#   inference(filePath)
#
## its that easy

#### Testing Purposes:
# if there is already a transcript file, delete it
if os.path.exists(transcript_path):
    os.remove(transcript_path)

file_index = 0
file_counter = 1
while isRunning:
    # reset audio path, then change it to the correct audio file path
    audio_path = home_dir + relative_path
    audio_path = audio_path+"output_{ind}.wav".format(ind=file_index)
    print(audio_path)

    # Wait until the audio file exists
    while not os.path.exists(audio_path):
        print(f"Waiting for {audio_path} to be created...")
        time.sleep(3.5)

    # Wait until the file is fully written by checking its size
    previous_size = -1
    while True:
        current_size = os.path.getsize(audio_path)
        if current_size == previous_size:
            break
        previous_size = current_size
        time.sleep(0.5)  # Wait a bit before checking again

    try:
        # Transcribe the audio file
        result = w_model.transcribe(audio_path, language="en")
        
        ## TO ADD CHECKING FOR SPEAKER HERE <<<<<
        #diarization = pipeline(audio_path)
        #print(diarization)


        
        # save text until 5 files have been added
        temp_transcript += result["text"].replace("'", "")
        #temp_transcript += speakerNameWithText.getSpeakerName() + ": " + speakerNameWithText.getText()
        # Print the transcription
        print(result["text"])

        if file_counter == 5:
            # i becomes the temp transcript, which stores the text for 5 files
            with open("speaker.txt", "r") as sp:
                speaker = sp.read()
            i =  speaker + temp_transcript

            topic = get_topic(history_topic, i)
            history_topic.append((i, topic))
            context = get_context_wiki(i)
            sufficiency = get_sufficiency(history_sufficiency, i, context).replace("```", "").replace("json", "")
            print(sufficiency)
            history_sufficiency.append((i, sufficiency))
            
            # Process the JSON string
            #fixed_json_str = remove_apostrophes_from_reason(sufficiency)

            # Print fixed JSON string
            #print(fixed_json_str)

            # Convert the fixed JSON string into a Python dictionary using json.loads
            try:
                read_sufficiency = json.loads(sufficiency)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")

            read_sufficiency['reason'] = read_sufficiency['reason'].replace("'", "")

            if read_sufficiency['sufficient_information'] == 'opinion' or read_sufficiency['sufficient_information'] == 'insignificant':

                fact_check_json = None #"factcheck: " + i + ": statement unclassifiable" 

            else:
                if read_sufficiency['sufficient_information'] == 'False':
                    context += get_context_news(topic)
                fc = factcheck(history_factcheck, i, context)
                history_factcheck.append((i, fc))
                fact_check_json = fc

            print(fact_check_json)
            with open(fact_checks_path, 'a') as file:
                if fact_check_json:
                    file.write(fact_check_json + '\n')

            # Open the file in append mode and save the transcription, then reset the counter and temp_transcription
            with open(transcript_path, 'a') as file:
                file.write(temp_transcript + '\n')
                temp_transcript = ""
                file_counter = 0

            # for ind in range(file_index - 5, file_index):
            #     ind =int(ind)
            #     file_to_delete = home_dir + relative_path + f"output_{ind}.wav"
            #     try:
            #         if os.path.exists(file_to_delete):
            #             os.remove(file_to_delete)
            #             print(f"Deleted file: {file_to_delete}")
            #         else:
            #             print(f"File not found: {file_to_delete}")
            #     except Exception as e:
            #         print(f"Error deleting file {file_to_delete}: {e}")

        # add text to total transcription
        total_transcript+=temp_transcript
        transcript_list.append(temp_transcript.split('.'))
        
    except Exception as e:
        print(f"Error processing file {audio_path}: {e}")

    # increment counter/index
    file_index += 1
    file_counter += 1
