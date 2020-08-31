# Image-Captioning
This model generates a caption for a image. This caption describes activities of input image.

Inspiration for this project came from blind peoples. I try to help those people with technology to some extent. Those people not able to see but they can hear! 
So, I build a project in which we attach a camera at a blind person's forehead or t-shirt which gives me a frame of an image and my Image Captioning model generates a caption for each image frame and used google text to speech API for  generate voice.

Here, this model is a combination of both CNN and RNN .
The architecture of model consists of two parts a) Encoder model and b)Decoder model
Encoder model take input a image and outputs the initial state of the Decoder model.  And Decoder model takes as inputs caption and output of the Encoder model as an initial state in GRUs and outputs the image caption.
