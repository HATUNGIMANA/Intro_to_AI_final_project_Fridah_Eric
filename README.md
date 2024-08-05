In this project, we use artificial intelligence to build a tool that helps visually impared people be able to better navigate the environment around them. We used two of the main leading technologies in the field of artificial intelligence: image recognition and natural language processing. 

Technical tools used include:
- Object Detection Algorithm - yolov7 Accessed from https://github.com/WongKinYiu/yolov7
- Image recognition and classification - ResNet50
- Natural Language Processing - GPT-2
- Text to speech model - gTTS
- Web platform - streamlit

Below is the step-to-step guide on how we implemented it and how anyone can have access to it and use it:

1. Imported necessary libraries such as:
          - cv2 - library used to process images
          - pyttsx3 - TTS library used to convert text to speech
          - threading - library used to increase speed by running tasks in separate threads.
          - flask - allows us to set up web interface
          - numpy - used to manage numerical arrays
          - tensorflow - used for object detection and text-to-speech
2. Loaded the Pre-trained models:
       - ResNet50 model for image classification
       - YOLOv7 for object detection

3. Created helper functions:
       - Image processing function
       - Object detection function
       - Caption generation function
       - Text-to-Speech function
4. Implementation of the streamlit web interface
       - Set up web interface using streamlit
       - Process the uploaded or captured image if applicable




  



