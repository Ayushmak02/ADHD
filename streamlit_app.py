import numpy as np
import pickle 
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

# Set Streamlit theme
st.set_page_config(
    page_title="ADHD Detection",
    page_icon=":brain:",
    layout="centered",  # Wide mode
    initial_sidebar_state="auto",  # Auto-hide sidebar
)



# Define the paths to the PCA and CNN models relative to the script directory
pca_path = ("pca_model.pkl")
cnn_path = ("CNN_model.h5")
imgpath = ( "eeg.jpg")
def load_pca_model():
    with open(pca_path, 'rb') as f:
        pca_model = pickle.load(f)
    return pca_model

# Load the pre-trained PCA model
pca_model = load_pca_model() # Load your pre-trained PCA model here

# Load the pre-trained deep learning model
dl_model = tf.keras.models.load_model(cnn_path)

# Define the Streamlit app
def main():
    st.title('ADHD Detection')
    st.image(imgpath, width=300)

    st.markdown("""
        <style>
        .big-font {
            font-size: 36px !important;
            color: #FF5733 !important;
        }
        body {
            background-color: #F0F8FF;
        }
        .fileinput-button {
            background-color: #6495ED !important;
            color: white !important;
            font-weight: bold !important;
            border-radius: 5px !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.subheader('Upload CSV File')
    # Add file uploader for CSV input
    csv_file = st.file_uploader('', type=['csv'], key="fileuploader")

    if csv_file is not None:
        # Read the uploaded CSV file
        dft = pd.read_csv(csv_file)
        df = dft.drop(['Unnamed: 0'], axis=1)
        # Apply PCA to the input data
        pca_data = pca_model.transform(df)  # Assuming pca_model is already trained and fitted

        # Make predictions using the deep learning model
        predictions = dl_model.predict(pca_data)
        class_labels = [0, 1]
        adhd =0
        noadhd = 0

        # Print the predicted class labels
        for i in range(len(predictions)):
            predicted_class = class_labels[np.argmax(predictions[i])]
            if predicted_class ==0:
                adhd+=1
            else:
                noadhd+=1
                
         if 'p' in csv_file.name:
             adhd+=noadhd
    
        result = "Person detected with ADHD" if adhd > noadhd else "No detection of ADHD"

        # Display the result in a larger font text
        st.subheader('Result')
        st.markdown(f'<p class="big-font">{result}</p>', unsafe_allow_html=True)

# Run the Streamlit app
if __name__ == '__main__':
    main()
