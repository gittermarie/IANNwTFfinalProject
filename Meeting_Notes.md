#### First Tutor Meeting
- Careful with literature (people don't know what they are doing)
- check citations and methods and people
- careful with training (overfitting) (maybe smaller network/small layers)
- bottlenecks
- use a pre-trained cnn (Resnets)
	- load trained model
	- good for problem with little training data
	- only fine tune it
- comparison between our own net and pre-trained net?
- make a project plan
	- include reading papers
	- know what we want to implement
- Segmentation or Classification?
- yolo-net -> segmentation 
	- https://pjreddie.com/darknet/yolo/
	- https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Redmon_You_Only_Look_CVPR_2016_paper.html
- mask r-cnn -> segmentation 
	- https://arxiv.org/abs/1703.06870

#### Meeting: 15.03. 17:00

### Comparison on Differnet Datasets between pretrained ResNet and Reimplemented CheXNet:
- Reimplement CheXNet (https://github.com/VinGPan/paper_implementations/blob/master/chest_xray_classification/CheXNet.py)
- Load Resnet (https://keras.io/api/applications/resnet/) and fine tune to Chest-X-ray dataset
- Compare both methods after period of training with validation data 
	- (F1? https://www.datasklr.com/select-classification-methods/model-selection)
- Then test both models performance on brain tumor or breast tumor data 
	- (using this as an example project? https://github.com/himanshu3997/Brain-tumor-MRI-Classification or this https://www.aimspress.com/article/doi/10.3934/mbe.2020328?viewType=HTML)

### Milestones:
1) Reimpliment CheXNet
2) Load and fine tune ResNet
3) Training
5) Run Chest-X-Ray Validation and Compare Results 
6) Run Brain Tumor Validation and Compare Results
7) Summarize Process and Results

##### Next Meeting: 20.03. 18:00
##### until then:
- Begin Reimplementing CheXNet (Marta)
- Begin Loading and fine tuning ResNet (Marie)
- Establish broad structure and begin documentation for final report (Lena)