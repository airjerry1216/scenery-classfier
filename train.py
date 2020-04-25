import torch
import torch.nn as nn
from models import VGG16
from dataset import IMAGE_Dataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
#import torchvision.models as models
#from torchvision import models
from torchvision import transforms
from pathlib import Path
import copy

##REPRODUCIBILITY
torch.manual_seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#args = parse_args()
#CUDA_DEVICES = args.cuda_devices
#DATASET_ROOT = args.path
CUDA_DEVICES = 1
#DATASET_ROOT = './seg_train'
DATASET_ROOT1 = '/home/lcc105u/Desktop/ccu_cars/train/'
PATH_TO_WEIGHTS = './model-best_train_acc.pth'

def train(i,train_acc,train_loss):
	data_transform = transforms.Compose([
		transforms.Resize((224,224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])
	#print(DATASET_ROOT)
	train_set = IMAGE_Dataset(Path(DATASET_ROOT1), data_transform)
	data_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True, num_workers=1)
	#print(train_set.num_classes)

	vgg16=models.vgg16(pretrained=True)		#載入vgg16 model
	pretrained_dict = vgg16.state_dict()	#vgg16預訓練的參數
	model_dict = VGG16.state_dict()			#自訂VGG16的參數

	# 將pretrained_dict裏不屬於model_dict的鍵剔除掉
	pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
	model_dict.update(pretrained_dict)	# 更新現有的model_dict
	VGG16.load_state_dict(model_dict)		# 加載我們真正需要的state_dict

	if i==1:
		model = VGG16(num_classes=train_set.num_classes)
	if i!=1:
		model=torch.load(PATH_TO_WEIGHTS)
	'''if i==1:
	    model = models.resnet101(pretrained=True)
	    fc_features=model.fc.in_features
	    model.fc=nn.Linear(fc_features,196)
	if i!=1:
	    model=torch.load(PATH_TO_WEIGHTS)'''
	model = model.cuda(CUDA_DEVICES)
	model.train()		#train

	best_model_params = copy.deepcopy(model.state_dict())    #複製參數
	best_acc = 0.0
	num_epochs = 10
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)

	for epoch in range(num_epochs):
		print(f'Epoch: {epoch + 1}/{num_epochs}')
		print('-' * len(f'Epoch: {epoch + 1}/{num_epochs}'))

		training_loss = 0.0
		training_corrects = 0

		for i, (inputs, labels) in enumerate(data_loader):
			inputs = Variable(inputs.cuda(CUDA_DEVICES))
			labels = Variable(labels.cuda(CUDA_DEVICES))			

			optimizer.zero_grad()

			outputs = model(inputs)
			_ , preds = torch.max(outputs.data, 1)
			loss = criterion(outputs, labels)

			loss.backward()
			optimizer.step()

			training_loss += loss.item() * inputs.size(0)
			#revise loss.data[0]-->loss.item()
			training_corrects += torch.sum(preds == labels.data)
			#print(f'training_corrects: {training_corrects}')
			#if(not(i%10)):
			#	print(f'iteration done :{i}')

		training_loss = training_loss / len(train_set)							#train loss
		training_acc =training_corrects.double() /len(train_set)				#tarin acc
		#print(training_acc.type())
		#print(f'training_corrects: {training_corrects}\tlen(train_set):{len(train_set)}\n')
		print(f'Training loss: {training_loss:.4f}\taccuracy: {training_acc:.4f}\n')
		train_acc.append(training_acc)		#save each 10 epochs accuracy
		train_loss.append(training_loss)

		if training_acc > best_acc:
			best_acc = training_acc
			best_model_params = copy.deepcopy(model.state_dict())

	model.load_state_dict(best_model_params)	#model load new best parms								#model載入參數
	torch.save(model, f'model-best_train_acc.pth')	#save model			#存整個model
	return(train_acc,train_loss)

#if __name__ == '__main__':
	#train()
