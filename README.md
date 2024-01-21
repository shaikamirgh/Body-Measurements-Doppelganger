# Body Measurements Computer Vision

![Demo](https://github.com/shaikamirgh/Body-Measurements-Doppelganger/OUTPUT/s7.jpg)
Do Check-Out OUTPUT folder for Results!

## Inspiration
*Most Returns of the orders on clothing (70%) are mainly due to size mismatches, while world has advanced in Technologies, why still the old way of "One Size Fits All?"*

## What it does
*Through Camera it takes Body Measurements and uses a Aruco marker for reference.*
Input- Camera Feed
Output- Accurate Body Measurements

## How we built it
The Project goes like this-
The Input Camera feed goes to the program, using Mediapipe library in python it detects Body Landmarks and then measures size of defined Aruco marker as reference point and finally
The measurements:
Head to Heel
Shoulder to Waist
Left shoulder to Right shoulder
Arm length 
are accurately derived and displayed for you to Choose your own best fit.
You can Also use these measurements to tailor cloth to your custom fit.
  
## Challenges we ran into
Camera doesn't find you Depth of the image, so initially we could not find whether the person is Taller or is he just too close of the camera.
To solve this we made a Aruco marker, took its Length and used it as a reference point which works by the math of a simple Cross multiplication i.e:
If Marker looks 10cm but is actually 20cm 
and Person looks 100cm then he is actually ___ cm which is 200!

## Accomplishments that we're proud of
We're Proud to have a High Accuracy of nearly 98% with precision of upto 1 cm!
Also given that the Project was done in a hackathon of a short duration.

## What we learned
We Learnt that as a Team with clear Goal even a Challenging task becomes Achievable in the given time limits.

## What's next for Doppleganger
Were now focusing on making a 3d clone for Body measurements on Blender using its API
Also we are planning to embed to into website through Browser Extensions so that a user has ease of access which Shopping Online.
[Github](https://github.com/shaikamirgh/Body-Measurements-Doppelganger)

Really Hoping that Our Team Wins!