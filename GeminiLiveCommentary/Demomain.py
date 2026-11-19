import gradio as gr
import time
import threading
import time
from google import genai
import os
from pathlib import Path
from dotenv import load_dotenv
# Set up Gemini client
# Load GEMINI_API_KEY from the .env file sitting next to this script.
# Explicit path so it works no matter which directory you run from.
load_dotenv(Path(__file__).with_name(".env"))

if not os.environ.get("GEMINI_API_KEY"):
    raise RuntimeError(
        "GEMINI_API_KEY is not set. Add it to the .env file next to this script "
        "(GEMINI_API_KEY=your-key-here), or set it as an environment variable."
    )
client = genai.Client()

# Function to upload video to Gemini
def upload_video_file(filename, poll_interval=2.0):
    try:
        print(f"Uploading: {filename}")
        myfile = client.files.upload(file=filename)

        while not myfile.state or myfile.state.name != "ACTIVE":
            if myfile.state and myfile.state.name == "FAILED":
                raise RuntimeError(
                    f"Gemini file processing failed: {myfile.name}"
                )

            state = myfile.state.name if myfile.state else "UNKNOWN"
            print(f"File state: {state}; waiting for ACTIVE...")

            time.sleep(poll_interval)
            myfile = client.files.get(name=myfile.name)

        print("File upload completed successfully")
        return myfile
    except Exception as e:
        print(f"Error uploading file: {str(e)}")
        raise

def GetResponse(prompt, myfile):
    response = client.models.generate_content_stream(
        model="gemini-3.5-flash-lite", contents=[myfile, prompt])
        #"You are an expert at Live Commentary for NBA games. Please give commentary in 75 words for the NBA game video provided and make sure to include the player names who are part of the game"]
    return response

def VideoDuration(video_file):
    # Get video duration
    clip = VideoFileClip(video_file)
    duration = clip.duration  # in seconds
    clip.close()
    return duration

def stream_text_Demo(video):
    video_file = video

    # Upload once and wait until Gemini has finished processing it
    uploaded_file = upload_video_file(video_file)

    # ONE prompt asking Gemini for all three outputs.
    # Therefore the video is passed to generate_content_stream only once.
    prompt = """
            You are an expert at live commentary for tennis games.

            Analyze the tennis video provided and produce ALL THREE of the following
            commentaries from the same video:

            1. A 225-word commentary in English.
            2. A 150-word commentary in English.
            3. A 75-word commentary in French.

            Make sure to include the names of the players who are part of the game
            when they can be identified from the video or audio.

            You MUST use exactly the following format:

            <<<COMMENTARY_225>>>
            [225-word English commentary]

            <<<COMMENTARY_150>>>
            [150-word English commentary]

            <<<COMMENTARY_FR_75>>>
            [75-word French commentary]

            Do not add any other sections or text outside these markers.
    """

    # ONLY ONE Gemini inference request
    stream = GetResponse(prompt, uploaded_file)

    # Full accumulated streamed response
    full_text = ""

    # Three Gradio outputs
    outputs = ["", "", ""]

    def extract_section(text, start_marker, end_marker=None):
        """
        Extract whatever has currently streamed between two markers.
        This lets each textbox update while Gemini is still generating.
        """

        if start_marker not in text:
            return ""

        section = text.split(start_marker, 1)[1]

        if end_marker is not None and end_marker in section:
            section = section.split(end_marker, 1)[0]

        return section.strip()

    # Stream the ONE response
    for chunk in stream:
        if hasattr(chunk, "text") and chunk.text:
            full_text += chunk.text

            outputs[0] = extract_section(
                full_text,
                "<<<COMMENTARY_225>>>",
                "<<<COMMENTARY_150>>>"
            )

            outputs[1] = extract_section(
                full_text,
                "<<<COMMENTARY_150>>>",
                "<<<COMMENTARY_FR_75>>>"
            )

            outputs[2] = extract_section(
                full_text,
                "<<<COMMENTARY_FR_75>>>"
            )

            # Update all three Gradio textboxes
            yield tuple(outputs)

## I think dynamic API is something worth trying just to check latency, but might not be optimal for our surface level, short clip reasoning. Long clips r good tho--Josh
# def stream_text_Demo(video):
#     video_file = video
#     uploaded_file = upload_video_file(video_file)
#     #duration = VideoDuration(video_file)
#     # Define three prompts (customize for your use-case)
#     prompts = [
#         f"You are an expert at Live Commentary for tennis games. Please give commentary in 225 words for the tennis game video provided and make sure to include the player names who are part of the game",
#         f"You are an expert at Live Commentary for tennis games. Please give commentary in 150 words for the tennis game video provided and make sure to include the player names who are part of the game",
#         f"You are an expert at Live Commentary for tennis games. Please give commentary in 75 words in french language for the tennis game video provided and make sure to include the player names who are part of the game."
#     ]

#     # Initialize empty outputs
#     outputs = ["", "", ""]

#     # Stream for each prompt separately
#     streams = [
#         GetResponse(prompt, uploaded_file) for prompt in prompts
#     ]

#     # Iterate over streamed chunks
#     while True:
#         finished = 0

#         for i, stream in enumerate(streams):
#             #import pdb;pdb.set_trace()
#             try:
#                 chunk = next(stream)
#                 if hasattr(chunk, "text") and chunk.text:
#                     outputs[i] += chunk.text
#                 yield tuple(outputs)
#             except StopIteration:
#                 finished += 1
#         if finished == len(streams):
#             break

"""  
if __name__ == "__main__":
    print("Test")
    stream_text_Demo()
    #print(outputs)
"""
with gr.Blocks() as demo:
    gr.Markdown("## 🎥 Closed Captioning Demo")

    video = gr.Video(label="Input Video", height=400)
    
    with gr.Row():
        t1 = gr.Textbox(label="225-Word Commentary", interactive=False, lines=10) 
        t2 = gr.Textbox(label="150-Word Commentary", interactive=False, lines=10)
        t3 = gr.Textbox(label="75-Word French Commentary", interactive=False, lines=10)

    btn = gr.Button("Start Analysis")

    # Use stream_text as generator for live updates
    btn.click(fn=stream_text_Demo, inputs=video, outputs=[t1, t2, t3])

demo.launch(share=True)
