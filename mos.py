import os
import tkinter as tk
from tkinter import simpledialog, messagebox
import pandas as pd
import pygame

# Path to the folder containing the audio files
audio_folder = r"C:\Users\Albin\Desktop\mos"

# Path to the output CSV file
output_csv = os.path.join(audio_folder, "mos_ratings.csv")

# Define the text for each audio file
audio_texts = {
    1: "Hello, welcome to our text-to-speech demo!",
    2: "The quick brown fox jumps over the lazy dog.",
    3: "Artificial intelligence makes life easier.",
    4: "Django is a powerful web framework for developers.",
    5: "Can you hear this text converted into speech?",
    6: "With text-to-speech technology, we can convert written words into lifelike audio.",
    7: "Speech synthesis allows computers to communicate in a way that feels more natural.",
    8: "This is a demonstration of how you can use Python to generate human-like speech.",
    9: "Text-to-speech systems are widely used in navigation devices and accessibility tools.",
    10: "The future of technology lies in making information accessible to everyone.",
    11: ("Text-to-speech technology is transforming the way we interact with devices."
         " By converting text into natural-sounding audio, it bridges the gap between written content and auditory experiences."
         " This innovation has applications in education, entertainment, and accessibility, making it easier for people with disabilities to engage with the digital world."),
    12: ("In recent years, artificial intelligence has significantly improved the quality of text-to-speech systems."
         " These advancements enable machines to produce speech that closely mimics human intonation, rhythm, and emotion."
         " As a result, TTS technology is now being integrated into various fields, including virtual assistants, audiobooks, and customer service."),
    13: ("Nature is a boundless source of inspiration and wonder. From the towering peaks of majestic mountains to the tranquil waves of the ocean, every element reflects an extraordinary diversity."
         " The chirping of birds at dawn, the rustling of leaves in the wind, and the gentle murmur of a flowing river create a harmonious symphony that soothes the soul."
         " Even in the smallest details, like the intricate patterns of a snowflake or the vibrant hues of a blooming flower, nature reminds us of the beauty that surrounds us."),
    14: ("Technology has transformed every aspect of our lives. It has made communication instant, learning accessible, and work more efficient."
         " From smartphones to self-driving cars, innovations continue to shape our future. However, these advancements come with challenges like data privacy and over-dependence."
         " As we embrace new technologies, it’s vital to use them responsibly to create a better, more inclusive world."),
    15: ("Perseverance is key to overcoming life’s challenges. It helps us push through difficulties and turn failures into learning experiences."
         " Many great inventors, athletes, and leaders have succeeded because they refused to give up, even when faced with obstacles."
         " Perseverance teaches us that success is not just about talent but also about dedication and the willingness to try again."),
}

# Initialize Tkinter
root = tk.Tk()

root.withdraw()  # Hide the main Tkinter window

try:
    # Initialize pygame mixer
    pygame.mixer.init()

    # List to store individual ratings
    individual_ratings = []

    # Check if the CSV file already exists
    if os.path.exists(output_csv):
        # If it exists, load the existing data to get previous ratings
        df_existing = pd.read_csv(output_csv)
        existing_participants = df_existing["Participant"].unique()
    else:
        # If no previous file, start with an empty set of participants
        existing_participants = []

    # Loop for multiple users
    while True:
        # Ask for the participant's name or ID
        participant_name = simpledialog.askstring(
            "Participant Info", "Enter your name or ID (or leave blank to finish):"
        )
        
        # If the user leaves the name blank, stop the process
        if not participant_name:
            break

        # Ensure the participant hasn't already rated
        if participant_name in existing_participants:
            messagebox.showinfo("Info", f"Participant {participant_name} has already rated. Skipping...")
            continue

        # Loop through each audio file and get a rating
        for idx in range(1, len(audio_texts) + 1):
            # Construct the audio file path
            audio_file = os.path.join(audio_folder, f"{idx}.wav")

            # Check if the audio file exists
            if not os.path.exists(audio_file):
                print(f"Audio file {idx}.wav not found. Skipping...")
                continue

            # Play the audio file using pygame
            print(f"Playing {audio_file}...")
            try:
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()

                # Wait until the audio finishes
                while pygame.mixer.music.get_busy():
                    pass
            except Exception as e:
                print(f"Error playing {audio_file}: {e}")
                continue

            # Show transcription and ask for rating
            transcription = audio_texts[idx]
            rating = simpledialog.askinteger(
                title=f"Rate Audio {idx}",
                prompt=(f"Audio: {idx}.wav\nTranscription: {transcription}\n\n"
                        "Rate the audio quality (1=Bad, 5=Excellent):"),
                minvalue=1,
                maxvalue=5
            )

            # If the user cancels the dialog, stop the process
            if rating is None:
                raise KeyboardInterrupt("Rating process canceled by user.")

            # Append the result for this participant
            individual_ratings.append((participant_name, f"{idx}.wav", transcription, rating))

    # Save the individual ratings to the CSV file
    if os.path.exists(output_csv):
        # Append to the existing file
        df_existing = pd.read_csv(output_csv)
        df_new = pd.DataFrame(individual_ratings, columns=["Participant", "Audio File", "Transcription", "Rating"])
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        # Create a new file
        df_combined = pd.DataFrame(individual_ratings, columns=["Participant", "Audio File", "Transcription", "Rating"])

    # Strip any extra spaces in column names to avoid issues
    df_combined.columns = df_combined.columns.str.strip()

    # Save to CSV
    df_combined.to_csv(output_csv, index=False)

    # Display success message
    messagebox.showinfo("Complete", f"Ratings have been saved to:\n{output_csv}")

except KeyboardInterrupt:
    # Handle user cancellation
    messagebox.showwarning("Interrupted", "The rating process was canceled.")
except Exception as e:
    # Handle unexpected errors
    messagebox.showerror("Error", f"An unexpected error occurred:\n{e}")
finally:
    # Quit pygame and Tkinter
    pygame.mixer.quit()
    root.destroy()

# Calculate the MOS (optional step after ratings are collected)
try:
    df = pd.read_csv(output_csv)
    mos_scores = df.groupby("Audio File")["Rating"].mean()
    mos_stddev = df.groupby("Audio File")["Rating"].std()

    # Calculate overall MOS and standard deviation
    overall_mos = mos_scores.mean()
    overall_stddev = mos_scores.std()

    print("\nMOS Scores per Audio File:")
    print(mos_scores)

    print(f"\nOverall MOS for the TTS system: {overall_mos:.2f} ± {overall_stddev:.2f}")
except Exception as e:
    print(f"Error calculating MOS: {e}")
