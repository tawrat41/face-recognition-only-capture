import streamlit as st  
import numpy as np
import os
import time
import cv2
import glob
import os
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

# Set the page configuration
st.set_page_config(
    page_icon=":smiley:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Function to capture image from webcam using OpenCV
def capture_and_save_image(label, save_folder):
    img_file_buffer = st.camera_input("Take Photo and then click Capture button again!")

    if img_file_buffer is not None:
        # Read image file buffer with OpenCV
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        # Save the image
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        image_name = f"{label}_captured_{time.time()}.png"
        image_path = os.path.join(save_folder, image_name)

        # Save the image to the specified folder
        cv2.imwrite(image_path, cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

        # Display the image
        st.image(cv2_img, caption=f"{label} Image")
        st.success(f"{label} Image captured and saved in {save_folder}")

def capture_and_save_image_test(label, save_folder):
    img_file_buffer = st.camera_input("Take Photo and then click Capture button again!")

    if img_file_buffer is not None:
        # Read image file buffer with OpenCV
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        # Save the image
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # Use a fixed name for the image to overwrite the existing one
        image_name = f"{label}_captured.png"
        image_path = os.path.join(save_folder, image_name)

        # Save the image to the specified folder
        cv2.imwrite(image_path, cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

        # Display the image
        st.image(cv2_img, caption=f"{label} Image")
        st.success(f"{label} Image captured and saved in {save_folder}")


session_state = st.session_state
if 'page_index' not in session_state:
    session_state.page_index = 0

# Function to get or create the session state
def get_session_state():
    if 'me_files' not in st.session_state:
        st.session_state.me_files = None
    if 'not_me_files' not in st.session_state:
        st.session_state.not_me_files = None



# Get or create session state
get_session_state()


# st.sidebar.title("Make your Face Recognition System")
st.sidebar.markdown("<h1>Make your Face Recognition System</h1>", unsafe_allow_html=True)
section = st.sidebar.radio("Steps to follow - ", ["Introduction", "Face Recognition by Computer", "Step - 1", "Collect Data", "Step - 2","Training Initiation", "Machine Learning", "Setup the Model", "Training Parameters", "Train", "Re-Train (if required)", "Step - 3","Test", "Improve Accuracy", "Step - 4", "Conclusion"],  index=session_state.page_index)

# Set the theme to light
st.markdown(
    """
    <style>
        body {
            font-family: 'Comic Sans MS', sans-serif !important;
            background-color: #fae0e4;  
        }

        p {
            text-align: justify;
            text-justify: inter-word;
            font-size: 1.2rem;
        }

        .center {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 0;
            margin-bottom: 10px;
        }
        .container{
            margin-left: 50px;
            margin-right: 50px;
        }

        h1{
            color:#ff0a54;
            text-align: center;
        }

        h2 {
            width: 100%;
            text-align: center;
            color: #ff0a54;
        }
        h4{
            color: #ff0a54;
        }
        h5{
            width: 100%;
            text-align: center;
        }

        .stButton button {
            background-color: white;
            color: #ff0a54;
            font-weight: bold #ff0a54;
            border: 2px solid ;
            border-radius: 5px;
            display: block;
            margin-left: auto;
            margin-right: auto;
            # width: 8rem;
            # text-align: center;
            padding-left: 15px;
            padding-right: 15px;
        }

        .stButton button:hover{
            background-color: #fbb1bd;
            color: white;
            border: 2px solid #fbb1bd;
        }

        #data-collect{
            font-size:2rem;
            font-weight:bold;
            color: #ff0a54  ;
            border: 2px solid #ff0a54;
            border-radius: 3px;
            padding: 2px 10px;
        }
        #collect-image p{
            color:#ff5c8a;
            font-weight: bold;
        }
        .stImage {
            display: flex;
            justify-content: center;
        }
        #MainMenu {
            color: #7209b7; /* Change the color code to the desired color */
        }
        .blank{
            height: 40px;
        }
        [data-testid="stSidebar"][aria-expanded="true"] {
            background-color: #f7cad0 !important; /* Change the color code to your desired color */
        }

        .streamlit-slider {
            color: #ff0a54 !important;
        }
            
    </style>
    """,
    unsafe_allow_html=True,
)


me_files = None
not_me_files = None



# Function to upload images
def upload_images(label, key):
    return st.file_uploader(f"Upload '{label}' images", type=["jpg", "png"], accept_multiple_files=True, key=key)

# Get or create session state
get_session_state()


# Section 1: What is Face Recognition? /////////////////////////////////////////////////////////////////////////////////////
if section == "Introduction":
    st.markdown('<div class="center"><h1>Face Recognition App</h1></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
                <div class=container><h2 class="header"> What is Face Recognition? </h2>
                     <p>Facial Recognition is a way of recognizing a human face using biometrics.It consists of comparing features of a person’s face with a database of known faces to find a match. When the match is found correctly, the system is said to have ‘recognized’ the face. Face Recognition is used for a variety of purposes, like unlocking phone screens, identifying criminals,
            and authorizing visitors. </p> </div>
            """, unsafe_allow_html=True)

    with col2:
    # st.markdown('<img src="media/Picture1.png">', unsafe_allow_html=True)
        image1 = Image.open('media/Picture1.png')
        st.image(image1, caption='')
    
    
    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
    col1, col2, col3= st.columns(3)

    with col1:
        pass
    with col3:
        pass
    with col2:
        st.button("Next", key="next_home", on_click=lambda: st.session_state.update({"page_index": 1}))    


    

# Face Recognition by Computer: Face Recognition by Computer /////////////////////////////////////////////////////////////////////////////////////
elif section == "Face Recognition by Computer":
    st.markdown('<div class="center"><h2>How do Computers Recognize Faces?</h2></div>', unsafe_allow_html=True)
    st.markdown("""
                <div class=container> <p>The Face Recognition system uses Machine Learning to analyze and process facial features from images or videos. Features can include anything, from the distance between your eyes to the size of your nose. These features, which are unique to each person, are also known as Facial Landmarks. The machine learns patterns in these landmarks by training Artificial Neural Networks. The machine can then identify people’s faces by matching these learned patterns against new facial data.
                    </p> </div>
                    """, unsafe_allow_html=True)


    col1, col2, col3 = st.columns([1, 2, 1])  # Adjust the width of col2 as needed

    with col1:
        pass

    with col2:
        video_url = "media/next_for_fr.mp4"
        st.video(video_url)

    with col3:
        pass




    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
    col1, col2= st.columns(2)
    with col1:
        st.button("Previous", key="prev_home", on_click=lambda: st.session_state.update({"page_index": 0}))
    with col2:
        st.button("Next", key="next_collect", on_click=lambda: st.session_state.update({"page_index": 2}))

    

elif section == "Step - 1":
    st.markdown("<div class='center'><h2>Teach the Computer to Recognize your Face</h2></div>", unsafe_allow_html=True)
    st.markdown("""
                <style>
                </style>""", unsafe_allow_html=True)

    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
    col0, col1, col2, col3, col4, col5, col6 = st.columns(7)
    with col0:
    # st.markdown("""
    #     <style>
    #     button[kind="primary"]  {
    #         background-color: white;
    #         color: grey;
    #         font-weight: bold #ff0a54;
    #         border: 2px solid ;
    #         display: block;
    #         margin-left: auto;
    #         margin-right: auto;
    #         # width: 8rem;
    #         # text-align: center;
    #         padding-left: 15px;
    #         padding-right: 15px;
    #     }
    #     </style>""", unsafe_allow_html=True)
        st.button("Step 1", key="step1", help="Collect Data", on_click=lambda: st.session_state.update({"page_index": 3}))
        st.markdown("""<p style="font-weight: bold;">Collect Data</p>""", unsafe_allow_html=True)
    with col1:
        st.markdown("""<p style="">&ndash;&ndash;&ndash;&ndash;&ndash;&ndash;&ndash;&#10148;</p>""", unsafe_allow_html=True)
    with col2:
        st.button("Step 2", key="step1-2", help="Train", on_click=None,disabled=True)
        st.markdown("""<p style="font-weight: bold;">Train</p> """, unsafe_allow_html=True)
    with col3:
        st.markdown("""<p style="">&ndash;&ndash;&ndash;&ndash;&ndash;&ndash;&ndash;&#10148;</p>""", unsafe_allow_html=True)
    with col4:
        st.button("Step 3", key="step3", help="Test", on_click=None,disabled=True)
        st.markdown("""<p style="font-weight: bold;">Test</p>    """, unsafe_allow_html=True)
    with col5:
        st.markdown("""<p style="">&ndash;&ndash;&ndash;&ndash;&ndash;&ndash;&ndash;&#10148;</p>""", unsafe_allow_html=True)
    with col6:
        st.button("Step 4", key="step4", help="Export", on_click=None,disabled=True)
        st.markdown("""<p style="font-weight: bold;">Export</p>    """, unsafe_allow_html=True)


    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
    col1, col2= st.columns(2)
    with col1:
        st.button("Previous", key="prev_capture", on_click=lambda: st.session_state.update({"page_index": 1}))
    with col2:
        st.button("Next", key="next_step-2", on_click=lambda: st.session_state.update({"page_index": 3}))


# Upload or Capture Image: Teach the Computer to Recognize your Face /////////////////////////////////////////////////////////////////////////////////////
elif section == "Collect Data":
    get_session_state()  # Get or create session state
    st.markdown("<div class = 'center'><h2 id='Section-3'>Step 1 - Collect Data</h2></div>", unsafe_allow_html=True)

    st.markdown("""
                <div class=container style="margin-bottom:20px;"> <p>We want our model to learn how to recognize your face. We will need two kinds of images for this - images of you, and images of people who are not you. This way, the model will learn to recognize how you look and also recognize how you don’t look. </p>  </div> """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
                <div class="container" id='collect-image'> <span id='data-collect'>1</span> <p>Let’s start by giving the machine lots of images of you in different places, in different poses, and at different angles. </p> </div>""", unsafe_allow_html=True)

        st.markdown(""" <h4 style="margin-top:15px">Capture some images of yourself</h4>  """, unsafe_allow_html=True)
        with st.form(key='me_form'):
            if st.form_submit_button("Capture 'me' Image"):
                capture_and_save_image('me', os.path.abspath('captured_images/me'))

    with col2:
        st.markdown("""
                    <div class="container" id='collect-image'><span id='data-collect'>2</span> <p>Next, let’s give it images of people that are not you, so the machine understands the difference.</p>
                </div>""", unsafe_allow_html=True)

        st.markdown(""" <h4 style="margin-top:15px">Capture some images of other people</h4>  """ , unsafe_allow_html=True)
        with st.form(key='not_me_form'):
            if st.form_submit_button("Capture 'not me' Image"):
                capture_and_save_image('not_me', os.path.abspath('captured_images/not_me'))

    # Display uploaded and captured images

    def list_images(folder_path):
        # Get a list of all files in the folder
        files = os.listdir(folder_path)
        # Filter out non-image files (you can customize this based on your file extensions)
        image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        return image_files

    # Display images of the user \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    # Specify the path to the folder containing your images
    folder_path = "captured_images/me"

    # Get the list of image files in the folder
    image_files = list_images(folder_path)

    # Display images in 4 columns
    num_columns = 6
    col_width = int(12 / num_columns)  # Divide the total width into equal parts

    for i in range(0, len(image_files), num_columns):
        st.markdown("<div class='center'><h2 id='Section-4'>Captured 'ME' Images</h2></div>", unsafe_allow_html=True)

        images_in_row = image_files[i:i + num_columns]
        cols = st.columns(num_columns)

        for col, image_file in zip(cols, images_in_row):
            image_path = os.path.join(folder_path, image_file)
            col.image(image_path, caption=image_file, use_column_width=True)

    file_path = 'captured_images/me/*.png'

    if glob.glob(file_path):
        if st.button("Delete 'ME' Images"):
                for filename in glob.glob('captured_images/me/*.png'):
                    os.remove(filename)
    else:
        print('Upload some images.')

    # Display images of other people \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    # Specify the path to the folder containing your images
    folder_path = "captured_images/not_me"

    # Get the list of image files in the folder
    image_files = list_images(folder_path)

    # Display images in 4 columns
    num_columns = 6
    col_width = int(12 / num_columns)  # Divide the total width into equal parts

    for i in range(0, len(image_files), num_columns):
        st.markdown("<div class='center'><h2 id='Section-4'>Captured 'NOT ME' Images</h2></div>", unsafe_allow_html=True)

        images_in_row = image_files[i:i + num_columns]
        cols = st.columns(num_columns)

        for col, image_file in zip(cols, images_in_row):
            image_path = os.path.join(folder_path, image_file)
            col.image(image_path, caption=image_file, use_column_width=True)
    if hasattr(st.session_state, 'not_me_files') and st.session_state.not_me_files:
        num_columns = 6  # Change this to 5 if you want 5 images in a row
        not_me_files = st.session_state.not_me_files
        row2 = st.columns(num_columns)
        for i, file in enumerate(not_me_files):
            row2[i % num_columns].image(file, caption="Other People's Image", use_column_width=True)

    file_path = 'captured_images/not_me/*.png'

    if glob.glob(file_path):
        if st.button("Delete 'Not ME' Images"):
                for filename in glob.glob('captured_images/not_me/*.png'):
                    os.remove(filename)
    else:
        print('Upload some images.')
    

    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)    
    with col1:
        st.button("Previous", key="prev_capture", on_click=lambda: st.session_state.update({"page_index": 2}))
    with col2:
        st.button("Next", key="next_step-2", on_click=lambda: st.session_state.update({"page_index": 4}))

# step - 2  /////////////////////////////////////////////////////////////////////////////////////
elif section == "Step - 2":
    st.markdown("<div class='center'><h2>Teach the Computer to Recognize your Face</h2></div>", unsafe_allow_html=True)
    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
    col0, col1, col2, col3, col4, col5, col6 = st.columns(7)

    with col0:
        st.button("Step 1", key="step1", help="Collect Data", on_click=lambda: st.session_state.update({"page_index": 3}))
        st.markdown("""<p style="font-weight: bold;">Collect Data</p>""", unsafe_allow_html=True)
    with col1:
        st.markdown("""<p style="">&ndash;&ndash;&ndash;&ndash;&ndash;&ndash;&ndash;&#10148;</p>""", unsafe_allow_html=True)
    with col2:  
        st.button("Step 2", key="step2", help="Train", on_click=lambda: st.session_state.update({"page_index": 5}))
        st.markdown("""<p style="font-weight: bold;">Train</p> """, unsafe_allow_html=True)
    with col3:
        st.markdown("""<p style="">&ndash;&ndash;&ndash;&ndash;&ndash;&ndash;&ndash;&#10148;</p>""", unsafe_allow_html=True)
    with col4:
        st.button("Step 3", key="step3", help="Test", on_click=None,disabled=True)
        st.markdown("""<p style="font-weight: bold;">Test</p>    """, unsafe_allow_html=True)
    with col5:
        st.markdown("""<p style="">&ndash;&ndash;&ndash;&ndash;&ndash;&ndash;&ndash;&#10148;</p>""", unsafe_allow_html=True)
    with col6:
        st.button("Step 4", key="step4", help="Export", on_click=None,disabled=True)
        st.markdown("""<p style="font-weight: bold;">Export</p>    """, unsafe_allow_html=True)

    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
    col1, col2= st.columns(2)
    with col1:
        st.button("Previous", key="prev_cap", on_click=lambda: st.session_state.update({"page_index": 3}))
    with col2:
        st.button("Next", key="next_train", on_click=lambda: st.session_state.update({"page_index": 5}))

# Training Initiation: Some other section 
elif section == "Training Initiation":
    # Use session state to access uploaded files
    get_session_state() 
    me_files = st.session_state.me_files
    not_me_files = st.session_state.not_me_files

    st.markdown('<div class="center"><h2 class="header"> Step 2 - Train The Machine </h2> </div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
                <div class=container> <p>Next, we need to train the machine (or model) to recognize pictures of you. The model uses the samples of images you provided for this. This method is called “Supervised learning” because of the way you ‘supervised’ the training. The model learns from the patterns in the photos you’ve taken. It mostly takes into consideration the facial features or Facial Landmarks and associates the landmark of each face with the corresponding label.
                </p> </div>  """, unsafe_allow_html=True)

    with col2:
        image1 = Image.open('media/Picture2.png')
        st.image(image1, caption='')

    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
    col1, col2= st.columns(2)
    with col1:
        st.button("Previous", key="prev_st-2", on_click=lambda: st.session_state.update({"page_index": 4}))
    with col2:
        st.button("Next", key="next_ml", on_click=lambda: st.session_state.update({"page_index": 6}))

# sectino 7 /////////////////////////////////////////////////////////////////////////////////////
elif section == "Machine Learning":
    st.markdown('<div class = "center"><h2 class="header"> What do you mean by machine learning ?</h2></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        
        st.markdown("""
                <div class=container> <p>Machine learning is the process of making systems that learn and improve by themselves. The model learns from the data and makes predictions. It then checks with your label to see if it predicted the label correctly. If it didn’t, then it tries again. It keeps repeating this process with an aim to get better at the predictions.
                </p> </div>  """, unsafe_allow_html=True)

    with col2:
    # st.markdown('<img src="media/Picture1.png">', unsafe_allow_html=True)
        image1 = Image.open('media/Picture4.png')
        st.image(image1, caption='')

    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
    col1, col2= st.columns(2)
    with col1:
        st.button("Previous", key="prev_ini", on_click=lambda: st.session_state.update({"page_index": 5}))
    with col2:
        st.button("Next", key="next_setup_model", on_click=lambda: st.session_state.update({"page_index": 7}))

# Setup the Model /////////////////////////////////////////////////////////////////////////////////////
elif section == "Setup the Model":
    st.markdown('<div class = "center"><h2 class="header">How to Setup the Model for Training ?</h2> </div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
                <div class=container> <p>Machine learning models can be of different types. One commonly used model is an Artificial Neural Network (ANN). 
                
                Neural networks mimic how the human brain works and interprets information. They consist of a many interconnected elements called Nodes. These nodes are organized in multiple layers, where nodes of each layer connect to nodes of the next layer.  

                </p> </div>  """, unsafe_allow_html=True)

    with col2:
    # st.markdown('<img src="media/Picture1.png">', unsafe_allow_html=True)
        image1 = Image.open('media/Picture5.png')
        st.image(image1, caption='')

    st.markdown('<div class="center"><h2 class="header">ANNs are Similar to our Brain </h2> </div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
                <div class=container> <p>The basic unit of our brain’s system is a neuron. We have approximately 86 billion neurons in our brain. Each is connected to another neuron through connections called synapses.

                A neuron can pass an electrical impulse to another neuron that is connected to it. This neuron can further pass on another electrical impulse to yet another connected neuron. In this way, a complex network of neurons is created in the human brain.

                This same concept (of a network of neurons) is used in machine learning or ANNs. In this case, the neurons are artificially created in a machine learning system. When many such artificial neurons (nodes) are connected, it becomes an Artificial Neural Network.


                </p> </div>  """, unsafe_allow_html=True)

    with col2:
    # st.markdown('<img src="media/Picture1.png">', unsafe_allow_html=True)
        image1 = Image.open('media/Picture6.gif')
        st.image(image1, caption='')

    st.markdown("""
                <div class=container> <p>Neurons in the first layer process signals that are input into the Neural network. They then send the results to connected neurons in the second layer. These results are then processed by neurons of the second layer and the results of this processing are sent to neurons of the third layer. This process continues till the signal reaches the last layer.

                The first layer of the Neural Network is called the input layer and, while the last layer is called the output layer. All layers in the middle comprise the hidden layers.

                </p> </div>  """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col1:
        pass

    with col2:
        image1 = Image.open('media/Picture7.gif')
        st.image(image1, caption='', width=None)

    with col3:
        pass
    


    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
    col1, col2= st.columns(2)
    with col1:
        st.button("Previous", key="prev_ml", on_click=lambda: st.session_state.update({"page_index": 6}))
    with col2:
        st.button("Next", key="next_para", on_click=lambda: st.session_state.update({"page_index": 8}))


# training parameters /////////////////////////////////////////////////////////////////////////////////////
elif section == "Training Parameters":
    st.markdown('<div class = "center"><h2 class="header">Training Parameters</h2> </div>', unsafe_allow_html=True)
    st.markdown("""
                <div class=container> <p>How your model trains depends on the Training Parameters that you set. Training parameters are values that control certain properties of the training process and of the resulting ML model. Let’s look at 2 important training parameters – epochs and learning rate, number of layers.
                </p> </div>  """, unsafe_allow_html=True)
    st.markdown("""  <h4>Epochs:</h4>  """, unsafe_allow_html=True)
    st.markdown("""
                <div class=container> <p>In machine learning, "epochs" are the number of times the algorithm goes through the entire training dataset. It's like repeating a book or a song multiple times to remember it better. More epochs give the machine learning model more opportunities to learn from the data, and it can become more accurate. 
                However, too many epochs can also make the model memorize the data instead of learning from it, which isn't good.
                </p> </div>  """, unsafe_allow_html=True)
    st.markdown("""  <h4>Learning rate:</h4>  """, unsafe_allow_html=True)
    st.markdown("""
                <div class=container> <p>Learning rate is how fast you want the model to learn during training. Think of it as how big a step the model takes when trying to improve itself. A high learning rate means big steps, and the model may overshoot the best solution. A low learning rate means small steps, and the model may take a long time to improve or may get stuck in a suboptimal solution. Finding the right learning rate is important because it affects how quickly and effectively the model learns.

                </p> </div>  """, unsafe_allow_html=True)
    st.markdown("""  <h4>Number of Hidden Layers: </h4>  """, unsafe_allow_html=True)
    st.markdown("""
                <div class=container> <p>Every Neural Network has one input and one output layer, but can have any number of hidden layers. Machine Learning Engineers often use systematic experimentation to discover what works best for the specific data. They train the model with a different number of hidden layers to see which one works best.
                </p> </div>  """, unsafe_allow_html=True)
    
    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
    col1, col2= st.columns(2)
    with col1:
        st.button("Previous", key="prev_set", on_click=lambda: st.session_state.update({"page_index": 7}))
    with col2:
        st.button("Next", key="next_train", on_click=lambda: st.session_state.update({"page_index": 9}))


# train /////////////////////////////////////////////////////////////////////////////////////
elif section == "Train":

    st.markdown(
    """
    <style>
        .stApp {
            # max-width: 800px;
            margin: auto;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True
    )
    

    # Use session state to access uploaded files
    me_files = st.session_state.me_files
    not_me_files = st.session_state.not_me_files

    st.markdown('<h2 class="header"> Train the Machine </h2> ', unsafe_allow_html=True)


    st.markdown(  """<p style="text-align: center;">Now let us set up our Machine Learning model! Enter the number of epochs for which you would like the model to train</p>""", unsafe_allow_html=True)


    st.markdown(  """<h5 style="text-align: center; margin-top:20px; color:#f72585">Number of epochs</h54>""", unsafe_allow_html=True)
    epochs_duplicate = st.slider("", 10, 100, 10)
    epochs = (epochs_duplicate // 10)
    st.markdown(  """<p style="text-align: center;">Once your model is all set, you can start training your model.</p>""", unsafe_allow_html=True)


    # Use session state to access uploaded files
    get_session_state() 
    me_files = st.session_state.me_files
    not_me_files = st.session_state.not_me_files

    # Train the model
    try:
        if st.button("Train Model"):
            # Gather paths for uploaded and captured images
            me_folder = os.path.abspath('captured_images/me')
            not_me_folder = os.path.abspath('captured_images/not_me')

            # Process uploaded images
            processed_images = []
            labels = []

            for uploaded_file in me_files or []:
                img = image.load_img(uploaded_file, target_size=(224, 224))
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                img = preprocess_input(img)
                processed_images.append(img)
                labels.append(1)  # 'me' class

            for uploaded_file in not_me_files or []:
                img = image.load_img(uploaded_file, target_size=(224, 224))
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                img = preprocess_input(img)
                processed_images.append(img)
                labels.append(0)  # 'not me' class

            # Process captured images if the folders exist
            if os.path.exists(me_folder) and os.path.exists(not_me_folder):
                for img_filename in os.listdir(me_folder):
                    img_path = os.path.join(me_folder, img_filename)
                    img = image.load_img(img_path, target_size=(224, 224))
                    img = image.img_to_array(img)
                    img = np.expand_dims(img, axis=0)
                    img = preprocess_input(img)
                    processed_images.append(img)
                    labels.append(1)

                for img_filename in os.listdir(not_me_folder):
                    img_path = os.path.join(not_me_folder, img_filename)
                    img = image.load_img(img_path, target_size=(224, 224))
                    img = image.img_to_array(img)
                    img = np.expand_dims(img, axis=0)
                    img = preprocess_input(img)
                    processed_images.append(img)
                    labels.append(0)

            if processed_images:  # Check if any images are available for training
                X_train = np.vstack(processed_images)
                y_train = np.array(labels)

                # Rest of your training code...
                st.markdown(f"<h5 style='text-align: center;'>Training with <span style='color:#ff0a54; font-size: 1.8rem;'>{epochs_duplicate} </span>epochs...</h4>", unsafe_allow_html=True)

                base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
                x = base_model.output
                x = GlobalAveragePooling2D()(x)
                x = Dense(1024, activation='relu')(x)
                predictions = Dense(2, activation='softmax')(x)  # 2 classes: 'me' and 'not me'
                model = Model(inputs=base_model.input, outputs=predictions)

                for layer in base_model.layers:
                    layer.trainable = False

                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                model.fit(X_train, y_train, epochs=epochs)  # Training with user-defined epochs

                # Calculate training accuracy
                train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)

                formatted_percentage = f"{train_acc * 100:.2f}%"
                st.markdown(f"<h4 style='text-align: center;'>Training complete! Training Accuracy: <span style='color:#ff0a54; font-size: 1.8rem; font-weight:bold'>{formatted_percentage}</span></h4>", unsafe_allow_html=True)

                # Save the model

                st.markdown(""" <h5 style="text-align:center;">A good accuracy is between 90-95%. If your model's accuracy is not good enough, click on the "Next" button, else click on the "Skip" button  </h5>""", unsafe_allow_html=True)

                model.save('model.h5')
            else:
                st.markdown(f"<h5 style='text-align: center; color:red'>No images available for training. Please capture or upload images.</h4>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred during training: {str(e)}")

    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("Previous", key="prev_para", on_click=lambda: st.session_state.update({"page_index": 8}))
    with col2:
        st.button("Skip", key="skip", on_click=lambda: st.session_state.update({"page_index": 11}))

    with col3:
        st.button("Next", key="next_re-train", on_click=lambda: st.session_state.update({"page_index": 10}))
       
# re train /////////////////////////////////////////////////////////////////////////////////////
elif section == "Re-Train (if required)":
    st.markdown(
    """
    <style>
        .stApp {
            # max-width: 800px;
            margin: auto;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True
    )
    

    # Use session state to access uploaded files
    me_files = st.session_state.me_files
    not_me_files = st.session_state.not_me_files

    st.markdown('<h2 class="header"> Train the Machine Again </h2> ', unsafe_allow_html=True)

    st.markdown(  """<p style="text-align: center;">If the accuracy is not good enough you can consider re-adjusting the training parameters, and training again</p>""", unsafe_allow_html=True)

    st.markdown(  """<h5 style="text-align: center; margin-top:20px; color:#f72585">Number of epochs</h54>""", unsafe_allow_html=True)
    epochs_duplicate = st.slider("", 10, 100, 10)
    epochs = (epochs_duplicate // 10)
    st.markdown(  """<p style="text-align: center;">Once your model is all set, you can start training your model.</p>""", unsafe_allow_html=True)


    # Use session state to access uploaded files
    get_session_state() 
    me_files = st.session_state.me_files
    not_me_files = st.session_state.not_me_files

    # Train the model
    if st.button("Train Model"):
        # Gather paths for uploaded and captured images
        me_folder = os.path.abspath('captured_images/me')
        not_me_folder = os.path.abspath('captured_images/not_me')

        # Process uploaded images
        processed_images = []
        labels = []

        for uploaded_file in me_files or []:
            img = image.load_img(uploaded_file, target_size=(224, 224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            processed_images.append(img)
            labels.append(1)  # 'me' class

        for uploaded_file in not_me_files or []:
            img = image.load_img(uploaded_file, target_size=(224, 224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            processed_images.append(img)
            labels.append(0)  # 'not me' class

        # Process captured images if the folders exist
        if os.path.exists(me_folder) and os.path.exists(not_me_folder):
            for img_filename in os.listdir(me_folder):
                img_path = os.path.join(me_folder, img_filename)
                img = image.load_img(img_path, target_size=(224, 224))
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                img = preprocess_input(img)
                processed_images.append(img)
                labels.append(1)

            for img_filename in os.listdir(not_me_folder):
                img_path = os.path.join(not_me_folder, img_filename)
                img = image.load_img(img_path, target_size=(224, 224))
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                img = preprocess_input(img)
                processed_images.append(img)
                labels.append(0)

        if processed_images:  # Check if any images are available for training
            X_train = np.vstack(processed_images)
            y_train = np.array(labels)

            # Rest of your training code...
        else:
            st.warning("No images available for training. Please capture or upload images.")

    
        # Train the model, save the model, etc.
        st.write(f"Training with {epochs_duplicate} epochs...")
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(2, activation='softmax')(x)  # 2 classes: 'me' and 'not me'
        model = Model(inputs=base_model.input, outputs=predictions)

        for layer in base_model.layers:
            layer.trainable = False

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=epochs)  # Training with user-defined epochs

        # Calculate training accuracy
        train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
        st.write(f"Training complete! Training Accuracy: {train_acc * 100:.2f}%")

        # Save the model
        model.save('model.h5')

    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
    col1, col2= st.columns(2)
    with col1:
        st.button("Previous", key="prev_train", on_click=lambda: st.session_state.update({"page_index": 9}))
    with col2:
        st.button("Next", key="next_st-3", on_click=lambda: st.session_state.update({"page_index": 11}))

# step - 3 /////////////////////////////////////////////////////////////////////////////////////
elif section == "Step - 3":
    
    st.markdown("<div class='center'><h2>Teach the Computer to Recognize your Face</h2></div>", unsafe_allow_html=True)
    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
    col0, col1, col2, col3, col4, col5, col6 = st.columns(7)

    with col0:
        st.button("Step 1", key="step1", help="Collect Data", on_click=lambda: st.session_state.update({"page_index": 3}))
        st.markdown("""<p style="font-weight: bold;">Collect Data</p>""", unsafe_allow_html=True)
    with col1:
        st.markdown("""<p style="">&ndash;&ndash;&ndash;&ndash;&ndash;&ndash;&ndash;&#10148;</p>""", unsafe_allow_html=True)
    with col2:
        st.button("Step 2", key="step2", help="Train", on_click=lambda: st.session_state.update({"page_index": 5}))
        st.markdown("""<p style="font-weight: bold;">Train</p> """, unsafe_allow_html=True)
    with col3:
        st.markdown("""<p style="">&ndash;&ndash;&ndash;&ndash;&ndash;&ndash;&ndash;&#10148;</p>""", unsafe_allow_html=True)
    with col4:
        st.button("Step 3", key="step3", help="Test", on_click=lambda: st.session_state.update({"page_index": 12}))
        st.markdown("""<p style="font-weight: bold;">Test</p>    """, unsafe_allow_html=True)
    with col5:
        st.markdown("""<p style="">&ndash;&ndash;&ndash;&ndash;&ndash;&ndash;&ndash;&#10148;</p>""", unsafe_allow_html=True)
    with col6:
        st.button("Step 4", key="step4", help="Export", on_click=None,disabled=True)
        st.markdown("""<p style="font-weight: bold;">Export</p>    """, unsafe_allow_html=True)

    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
    col1, col2= st.columns(2)
    with col1:
        st.button("Previous", key="prev_st-3", on_click=lambda: st.session_state.update({"page_index": 10}))
    with col2:
        st.button("Next", key="next_test", on_click=lambda: st.session_state.update({"page_index": 12}))

# test /////////////////////////////////////////////////////////////////////////////////////
elif section == "Test":
    st.markdown('<h2 class="header"> Test the model </h2> ', unsafe_allow_html=True)
    st.markdown("""
                <style>
                    #test-span{
                        color: #ff0a54;
                        font-weight: bold;
                }
                </style>


        """, unsafe_allow_html=True)



    st.markdown("""
                <div class=container style="margin-bottom:20px;"> <p>
                It's time to put the model to the test. You can evaluate its performance by either <span id="test-span">uploading</span> an image or <span id="test-span">capturing</span> one in real-time. This testing phase will help you assess the model's capabilities in handling visual data, giving you valuable insights into its effectiveness. Choose the method that suits your evaluation preferences, whether it's uploading a pre-existing image or utilizing the model's image-capturing feature for a more dynamic experience.</p>  </div> """, unsafe_allow_html=True)


    col1, col2, col3 = st.columns(3)

    with col1:
        # # Option to upload a test image
        # st.markdown(""" <h4>Upload Test Image</h4>  """, unsafe_allow_html=True)
        # test_image = st.file_uploader("", type=["jpg", "png"])
        pass

    with col2: 
        # Option to capture a test image
        st.markdown(""" <h4 style="margin-bottom:30px">Capture Test Image</h4>  """ , unsafe_allow_html=True)
        with st.form(key='test_form'):
            if st.form_submit_button("Capture test Image"):
                capture_and_save_image_test('test', os.path.abspath('captured_images/test_capture'))
            
        def list_images(folder_path):
            # Get a list of all files in the folder
            files = os.listdir(folder_path)
            # Filter out non-image files (you can customize this based on your file extensions)
            image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
            return image_files
        folder_path = 'captured_images/test_capture'
        image_files = list_images(folder_path)


    with col3:
        pass


    

    # Process the uploaded or captured test image
    if st.button("Test Model"):
        # if test_image or os.path.exists('captured_images/test_capture'):     //////////// comment out if upload option is enabled
        if os.path.exists('captured_images/test_capture'):
            st.markdown(f"<h5 style='text-align: center;'>Processing test image...</h5>", unsafe_allow_html=True)
            
            if os.path.exists('captured_images/test_capture'):
                # Use the captured test image
                test_image_path = os.path.join('captured_images/test_capture', os.listdir('captured_images/test_capture')[0])
            # else:
                # # Use the uploaded test image
                # test_image_path = 'uploaded_test_image.png'
                # with open(test_image_path, "wb") as f:
                #     f.write(test_image.read())

            img = image.load_img(test_image_path, target_size=(224, 224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)

            # Make prediction
            try:
                model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
                x = model.output
                x = GlobalAveragePooling2D()(x)
                x = Dense(1024, activation='relu')(x)
                predictions = Dense(2, activation='softmax')(x)  # 2 classes: 'me' and 'not me'
                model = Model(inputs=model.input, outputs=predictions)

                model.load_weights('model.h5')
                prediction = model.predict(img)
                predicted_class = np.argmax(prediction)

                # Display result
                if predicted_class == 1:
                    # st.write("Result: This is you!")
                    st.markdown(f"<h4 style='text-align: center;'>Result: This is you!</h4>", unsafe_allow_html=True)

                else:
                    # st.write("Result: This is not you.")
                    st.markdown(f"<h4 style='text-align: center;'>Result: This is not YOU!</h4>", unsafe_allow_html=True) 

            except FileNotFoundError as e:
                # st.warning(str(e))
                st.markdown(""" <h5 style="color:red;">Trained model not found! Train the Model first.</h5> """, unsafe_allow_html=True)

        else:
            st.markdown(""" <h5 style="color:red;">Please upload a test image or capture one.</h5> """, unsafe_allow_html=True)

            
    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
    col1, col2= st.columns(2)
    with col1:
        st.button("Previous", key="prev_st-3", on_click=lambda: st.session_state.update({"page_index": 11}))
    with col2:
        st.button("Next", key="next_acc", on_click=lambda: st.session_state.update({"page_index": 13}))

# Improve Accuracy /////////////////////////////////////////////////////////////////////////////////////
elif section == "Improve Accuracy":

    st.markdown('<h2 class="header" style="text-align:center;"> Was the model able to recognize the face correctly?</h2>', unsafe_allow_html=True)
    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 3, 1])

    with col1:
        pass

    with col2:
        # Buttons for "YES" and "NO"
        col4, col5 = st.columns(2)
        yes_button = col4.button("YES", key="yes_button")
        no_button = col5.button("NO", key="no_button")

        if yes_button:
            st.markdown('<h2 class="header">Great! Go to the next section.</h2>', unsafe_allow_html=True)

        if no_button:
            st.markdown('<h2 class="header"></h2>', unsafe_allow_html=True)
            col1, col2 = st.columns([1,1.5])
            with col1:
                st.markdown('<h4 class="header">If not, then you have 2 options: </h4>', unsafe_allow_html=True)
                st.markdown("""
                    <div style="margin-bottom: 10px;">
                        <ul>
                            <li>Adjust training parameters and Re-train the model</li>
                            <li>Add more images to your dataset</li>
                        </ul>
                            
                    </div>
                """, unsafe_allow_html=True)

            st.button("Upload or Capture more image", key="step1-again", help="Go to Step-1", on_click=lambda: st.session_state.update({"page_index": 3}))
            st.button("Adjust Training Parameters", key="Adjust", help="Go to Step-2", on_click=lambda: st.session_state.update({"page_index": 9}))


            with col2:
                st.markdown('<h4 class="header">How do you improve the accuracy of the system? </h4>', unsafe_allow_html=True)
                st.markdown("""
                    <div>
                        <ol>
                            <li>You can collect more data to train the system. The more data you feed into the system, the more exposed it is to a variety of examples and the better the predictions. Try training the model with more images of each label.</li>
                            <li>You can fine-tune the parameters of the machine learning model. Play around with the number of epochs, hidden layers, and the learning rate. Try different values and see which combination of parameters makes the model’s predictions more accurate.</li>
                        </ol>
                    </div>
                """,  unsafe_allow_html=True)

    with col3:
        pass

    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)

    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
    col1, col2= st.columns(2)
    with col1:
        st.button("Previous", key="prev_test", on_click=lambda: st.session_state.update({"page_index": 12}))
    with col2:
        st.button("Next", key="next_st-4", on_click=lambda: st.session_state.update({"page_index": 14}))

# step - 4 /////////////////////////////////////////////////////////////////////////////////////
elif section == "Step - 4":
    st.markdown("<div class='center'><h2>Teach the Computer to Recognize your Face</h2></div>", unsafe_allow_html=True)
    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
    col0, col1, col2, col3, col4, col5, col6 = st.columns(7)

    with col0:
        st.button("Step 1", key="step1", help="Collect Data", on_click=lambda: st.session_state.update({"page_index": 3}))
        st.markdown("""<p style="font-weight: bold;">Collect Data</p>""", unsafe_allow_html=True)
    with col1:
        st.markdown("""<p style="">&ndash;&ndash;&ndash;&ndash;&ndash;&ndash;&ndash;&#10148;</p>""", unsafe_allow_html=True)
    with col2:
        st.button("Step 2", key="step2", help="Train", on_click=lambda: st.session_state.update({"page_index": 5}))
        st.markdown("""<p style="font-weight: bold;">Train</p> """, unsafe_allow_html=True)
    with col3:
        st.markdown("""<p style="">&ndash;&ndash;&ndash;&ndash;&ndash;&ndash;&ndash;&#10148;</p>""", unsafe_allow_html=True)
    with col4:
        st.button("Step 3", key="step3", help="Test", on_click=lambda: st.session_state.update({"page_index": 12}))
        st.markdown("""<p style="font-weight: bold;">Test</p>    """, unsafe_allow_html=True)
    with col5:
        st.markdown("""<p style="">&ndash;&ndash;&ndash;&ndash;&ndash;&ndash;&ndash;&#10148;</p>""", unsafe_allow_html=True)
    with col6:
        st.button("Step 4", key="step4", help="Export", on_click=None)
        st.markdown("""<p style="font-weight: bold;">Export</p>    """, unsafe_allow_html=True)


    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
    col1, col2= st.columns(2)
    with col1:
        st.button("Previous", key="prev_st-4", on_click=lambda: st.session_state.update({"page_index": 13}))
    with col2:
        st.button("Next", key="next_con", on_click=lambda: st.session_state.update({"page_index": 15}))

# conclusion /////////////////////////////////////////////////////////////////////////////////////
elif section == "Conclusion":
    st.markdown("""<h2 class="header"> Congratulations! </h2> <h5>You just created your very own Face Recognition system! </h5>
    """, unsafe_allow_html=True)

    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
    col1, col2, col3= st.columns(3)
    with col1:
        pass
    with col3:
        pass
    with col2 :
        st.button("Previous", key="prev_st-4", on_click=lambda: st.session_state.update({"page_index": 14})) 
        st.button("Start Over?", key="start-over", on_click=lambda: st.session_state.update({"page_index": 0})) 

