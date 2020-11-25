from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import sys
sys.path.append('../')
#import openvino_demo_224

def one(request):
	if request.is_ajax():
		f = open('demofile.txt') #r = reading
		counter = f.read()
		f.close 
		#f = open("score.txt", "r")
		#score = f.read()
		f.close
		#return JsonResponse({'count':counter, 'score':score})
		return JsonResponse({'count':counter})

	else:
		return render(request, 'blog/one.html')

def two(request):
	if request.is_ajax():
		f = open('demofile.txt') #r = reading
		counter = f.read()
		f.close 
		f = open("score.txt", "r")
		score = f.read()
		f.close
		return JsonResponse({'count':counter, 'score':score})
		#return JsonResponse({'count':counter})

	else:
		return render(request, 'blog/two.html')

def three(request):
	if request.is_ajax():
		f = open('demofile.txt') #r = reading
		counter = f.read()
		f.close 
		f = open("score.txt", "r")
		score = f.read()
		f.close
		return JsonResponse({'count':counter, 'score':score})
		#return JsonResponse({'count':counter})

	else:
		return render(request, 'blog/three.html')

def four(request):
	if request.is_ajax():
		f = open('demofile.txt') #r = reading
		counter = f.read()
		f.close 
		f = open("score.txt", "r")
		score = f.read()
		f.close
		return JsonResponse({'count':counter, 'score':score})
		#return JsonResponse({'count':counter})

	else:
		return render(request, 'blog/four.html')

def openpose(request):
	#openvino_demo_224.main()
	return render(request, 'blog/about.html')

	
def home(request):
    
    return render(request, 'blog/home.html')


def about(request):
	if request.is_ajax():
		f = open('a.txt') #r = reading
		a = f.read()
		f.close 
		f = open("b.txt", "r")
		b = f.read()
		f.close
		f = open("c.txt", "r")
		c = f.read()
		f.close
		f = open("d.txt", "r")
		c = f.read()
		f.close
		f = open("e.txt", "r")
		c = f.read()
		f.close
		f = open("f.txt", "r")
		c = f.read()
		f.close
		return JsonResponse({'a':a, 'b':b, 'c':c, 'd':d,'e':e,'f':f})
		#return JsonResponse({'count':counter})

	else:
		return render(request, 'blog/about.html')
    

def abc(request):
	return render(request, 'blog/home.html')
