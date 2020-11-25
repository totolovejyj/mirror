from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import sys
sys.path.append('../')
import openvino_demo_224

def one(request):
	if request.is_ajax():
		f = open('demofile.txt') #r = reading
		counter = f.read()
		f.close 
		f = open("score.txt")
		score = f.read()
		f.close
		return JsonResponse({'count':counter, 'score':score})
		

	else:
		return render(request, 'blog/one.html')

def two(request):
	if request.is_ajax():
		f = open('demofile.txt') #r = reading
		counter = f.read()
		f.close 
		f = open("score.txt")
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
		f = open("score.txt")
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
		f = open("score.txt")
		score = f.read()
		f.close
		return JsonResponse({'count':counter, 'score':score})
		#return JsonResponse({'count':counter})

	else:
		return render(request, 'blog/four.html')

def openpose(request):
	a = openvino_demo_224.main()
	print("*********************")
	print(a)
	if a == 0:
		return render(request, 'blog/about.html')
	else:
		return

	
def home(request):
    return render(request, 'blog/home.html')


def about(request):
	if request.is_ajax():
		#f = open('a.txt') #r = reading
		#a = f.read()
		#f.close 
		#f = open("b.txt")
		#b = f.read()
		#f.close
		#f = open("c.txt")
		#c = f.read()
		#f.close
		f = open("max_angle.txt")
		max_angle = f.read()
		f.close
		#f = open("e.txt")
		#e = f.read()
		#f.close
		#f = open("f.txt")
		#f = f.read()
		#f.close
		#return JsonResponse({'a':a, 'b':b, 'c':c, 'max_angle':max_angle,'e':e,'f':f})
		return JsonResponse({'max_angle':max_angle})
		#return JsonResponse({'count':counter})

	else:
		return render(request, 'blog/about.html')
    

def abc(request):
	return render(request, 'blog/home.html')
