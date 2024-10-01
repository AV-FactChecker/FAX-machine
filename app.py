import streamlit as st
import pandas as pd
import subprocess
import os
import psutil
import time
import ast

# Get location of folders and files
home_dir = os.path.expanduser("~")
relative_path = "\\OneDrive\\Desktop\\AudioExtractor\\AudioExtractor.exe"
AudioExtractor_path = home_dir + relative_path
AudioExtractor_ZIP = "AudioRecorder.exe"
transcript_path = "transcript.txt"
fact_checks_path = "fact_checks.txt"

# Title of the app
st.title("FAX Machine")

with st.expander("Click Here to View/Hide Instructions"):
    st.write("First we need to be able to collect audio. To do this, download the following zip file.")
    
    # Read the zip file as binary
    with open(AudioExtractor_path, "rb") as file:
        zip_data = file.read()

    # Create a download button for the zip file
    st.download_button(label="Download Audio Recorder", data=zip_data, file_name="AudioRecorder.exe", mime="application/exe")

    st.write("Once it has been downloaded, right click on the .zip file in File Explorer, and hit \"Extract All\"")
    st.image("Images/Extract.jpg", caption="Extract All is indicated by the red box.")
    st.write("On the popup, hit browse, then navigate to your desktop, and hit \"Select Folder\"")
    st.image("Images/Browse.jpg")
    st.image("Images/SelectFolder.jpg")
    st.write("Lastly, hit \"Extract\" on the first popup, and you should now see a folder called AudioExtractor on your desktop with AudioExtractor.exe and a bunch of other files in it!")
    st.write("First, run the AudioExtractor, then run the speechtotext.py script.")
speaker = ""
with st.container():
    # Button to start displaying the text file content
    start_monitoring_dt = st.button("View Transcript/Fact Checks For Tim Walz")
    start_monitoring_kh = st.button("View Transcript/Fact Checks For JD Vance")

    # Initialize monitoring state if not already
    if 'monitoring' not in st.session_state:
        st.session_state.monitoring = False

    # Start monitoring when the button is pressed
    if start_monitoring_dt:
        st.session_state.monitoring = True
        speaker = "JD Vance: "
    if start_monitoring_kh:
        st.session_state.monitoring = True
        speaker = "Tim Walz: "
    with open("speaker.txt", "w") as spkfile:
            spkfile.write(speaker)

    # Create the "Stop Monitoring" button outside the loop
    if st.session_state.monitoring:
        stop_monitoring = st.button("Stop Monitoring")

    st.header("Live Transcript")
    st.write("Note: Random periods and minor inaccuracies are to be expected.")

    # Placeholder for the text file content
    text_display = st.empty()
    st.header("Fact Checks")
    st.write("Anytime the AI deems a statement to be a claim of fact, it will be checked by the AI and shown below. It will show the statement being checked, whether it is true or false, and the reasoning for the AI's determination. It will be updated everytime a statement is checked.")
    fact_display = st.empty()

    factstatementlist = []

    # While loop to monitor the file content
    if st.session_state.monitoring:
        while st.session_state.monitoring:
            if os.path.exists(transcript_path):
                with open(transcript_path, "r") as file:
                    # walmart text wrapping
                    content = file.read()
                    words = content.split()
                    result = []
                    for i in range(0, len(words), 10):
                        result.append(' '.join(words[i:i+10]))

                    content = '\n'.join(result)

                # Update the placeholder with the current content
                text_display.text(content)

            if os.path.exists(fact_checks_path):
                with open(fact_checks_path, "r") as file:
                    content = file.read()   #.replace("{","").replace("}","").replace("'", "") + "\n\n"
                    factchecklist = content.split("}")
                    # st.write(factchecklist)
                    factchecks = ""
                    
                    for f in factchecklist:
                        if not factchecklist.index(f) == len(factchecklist)-1:
                            f += "}"
                        else:    
                            f +=  "{\"statement\": \"\",\"result\": True, \"reason\": \"\"}"
                        
                        # st.write(factstatementlist)

                        factdict = ast.literal_eval(f)
                        factdictresult = factdict["result"]
                        # st.write("before false check")
                        if factdictresult == False:
                            factchecks += "FAKE NEWS ALERT: " + factdict["statement"] + "\n" + "REASON: " + factdict["reason"] + "\n"
                            if factdict["statement"] not in factstatementlist:
                                factcheck_dataframe = pd.DataFrame({
                                    "SPEAKER" : [factdict["speaker"]],
                                    "STATEMENT" : [factdict["statement"]],
                                    "REASON" : [factdict["reason"]]
                                })
                                st.dataframe(factcheck_dataframe)

                            # st.write("help me")
                            factstatementlist.append(factdict["statement"])
                    
                    # fact_display.text(factchecks)

                    # content = factchecks
                    # words = content.split()
                    # result = []
                    # for i in range(0, len(words), 10):
                    #     result.append(' '.join(words[i:i+10]))
                    # content = '\n'.join(result)
                    # fact_display.text(content)
                    
            # Check if the stop button was pressed
            if stop_monitoring:
                # st.error("help i made an error!")
                st.session_state.monitoring = False
                break

            # Sleep for 5 seconds to avoid excessive resource use
            time.sleep(5)

            

with st.sidebar:
    # Section for running an exe file
    st.header("Run the Audio Extractor")

    exe_file_path = st.text_input("Enter the path to the .exe file.", value=AudioExtractor_path)
    st.write("This button only works if the app is being run locally. If not, manually run the .exe file you downloaded. See instructions for more detail. Make sure that the username is changed to your current user.")
    exe = "AudioExtractor.exe"

    if st.button("Run Audio Extractor"):
        if os.path.exists(exe_file_path):
            try:
                process = subprocess.Popen([exe_file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                st.text("Audio Extractor is running...")
            except Exception as e:
                st.error(f"Error running executable: {e}")
        else:
            st.error("The specified file does not exist.")

    if st.button("Stop Audio Extractor"):
        try:
            for proc in psutil.process_iter():
                # check whether the process name matches
                if proc.name() == exe:
                    proc.kill()
                    st.text("Audio Extractor has been stopped.")
        except Exception as e:
            st.error(f"Error stopping executable: {e}")

    # Section for running a Python script
    st.header("Run Transcription Script")

    python_script_path = st.text_input("This script will transcribe any audio recorded on your device.", value="speechtotext.py")
    # st.write("Note: You may need to press the button twice to work. Look for text notifying if the script has been ran or stopped.")

    # Initialize session state for the button and script PID if not already
    if 'button_clicked' not in st.session_state:
        st.session_state.button_clicked = False
    if 'script_pid' not in st.session_state:
        st.session_state.script_pid = None

    # Code that runs the Python script
    def run_script():
        if os.path.exists(python_script_path):
            try:
                result = subprocess.Popen(["python", python_script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                st.session_state.script_pid = result.pid
                st.session_state.button_clicked = True
                st.text("Python script is running...")
            except Exception as e:
                st.error(f"Error running Python script: {e}")
        else:
            st.error("The specified file does not exist.")

    # Code that stops the Python script
    def stop_script():
        if st.session_state.script_pid:
            try:
                process = psutil.Process(st.session_state.script_pid)
                process.terminate()
                st.session_state.button_clicked = False
                st.text("Python script stopped.")
            except Exception as e:
                st.error(f"Error stopping Python script: {e}")
        else:
            st.error("No script is currently running.")

    # Show the Run button and Stop button based on the state
    if not st.session_state.button_clicked:
        if st.button("Run/Kill Script"):
            run_script()
    else:
        if st.button("Run/Kill Script"):
            stop_script()
