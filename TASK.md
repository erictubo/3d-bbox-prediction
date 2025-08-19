# Assignment

### Coding Challenge – 3D Bounding Boxes

#### *Deadline*

You will have 7 days to complete the challenge.

#### Task

Develop a deep learning pipeline (model + code infrastructure) for 3D bounding box prediction problem

#### Requirements

- Preferably written in Pytorch, but other DL frameworks can also be used. Utils libs such as albumentations, kornia, ... can also be adopted.
- End-to-end fully functional pipeline: Preprocessing -> data loading -> model + train loop -> test loop -> Inference optimization (optional, but with be a big bonus point if candidate can convert models into lower-precision format or universally deployable format such as ONNX or TensorRT).
- Brief documentation of how the candidates choose the architecture/ loss function to tackle the problem or how the code works (diagram will be nice here also).
- Candidates can choose their own metrics to measure the performance of the model.
- Showing the training and testing logs as well as output predictions visualization to verify the approach.
- Using high-level libraries like MMDetection or Ultralytics are permitted, though complete reliance on them will prevent candidates from demonstrating their own reasoning and coding abilities.

#### Data
- RGB, ground truth 3d bounding box, point cloud and instance segmentation mask.

#### Note
The test does not expect the candidate to fulfil all the requirements since this is a fairly hard problem given the limited data and time budget. What is important is the candidate showing competent coding skill with deep learning framework as well as his/her approach to a complex technical problem.

#### Data Link

https://drive.google.com/file/d/11s-GLb6LZ0SCAVW6aikqImuuQEEbT_Fb/view?usp=sharing