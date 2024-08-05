In this project, we use artificial intelligence to build a tool that helps visually impared people be able to better navigate the environment around them. We used two of the main leading technologies in the field of artificial intelligence: image recognition and natural language processing. 

Technical tools used include:
- Object Detection Algorithm - yolov7 Accessed from https://github.com/WongKinYiu/yolov7
- Image recognition and classification - ResNet50
- Natural Language Processing - GPT-2
- Text to speech model - gTTS
- Web platform - streamlit

Below is the step-to-step guide on how we implemented it and how anyone can have access to it and use it:

1. Imported necessary libraries such as:
- numpy - used to manage numerical arrays and matrices
- pandas - used to manipulate data
- PIL - used to work with the images
- TensorFlow - used to build and train neural networks
- ResNet50 - used to classify images
- Streamlit - used to create the web interface
- gtts - used to convert the text to speech audio files
- opencv - used to process the image and video analysis
  
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


To host this application on a local server you need to follow the following steps:
1. Install necessary dependancies
2. Create a streamlit app
3. Run the streamlit app using: streamlit run app.py
4. Implement Object and classification
5. Customize and enhance

Find here the video demo of his this application works: 

  



