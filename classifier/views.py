from django.shortcuts import render, redirect

# Create your views here.

from .models import *
from .load import classify
from .performance import performance_graph

def home(request):

	if request.method=="POST":
		files= request.FILES.getlist('image')
		imagelist=[]

		for file in files:
			image=Image()
			image.save()
			image.img= file
			image.save()
			imagelist.append(str(image.img))

		pred, imagepaths=classify(imagelist)
		mylist= zip(pred, imagepaths)
			
		context = {'mylist': mylist}
		return render(request,'classifier/classify.html', context)
			
	return render(request, 'classifier/front.html')


def performance(request):

	performance_graph()

	return render(request,'classifier/front.html')