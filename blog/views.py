from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
import sys
sys.path.append('../')
import openvino_demo_224
import test

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
		f = open('demofile.txt', 'w') #r = reading
		f.write(str(0))
		f.close 
		f = open("score.txt", 'w')
		f.write(str(0))
		f.close
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
	f = open('test.txt') #r = reading
	threshold = f.read()
	f.close 
	openvino_demo_224.main(threshold)
	#if request.is_ajax():
	#	print("openpose Ajax")
	#	return JsonResponse({'count': 100, 'score': 200})
	return render(request, 'blog/about.html')
#	return HttpResponseRedirect('/about')

def test(request):
	#openvino_demo_224.main()
	test()
	return render(request, 'blog/one.html')

	
def home(request):
    return render(request, 'blog/home.html')


def about(request):
	if request.is_ajax():
		f = open('time.txt') #r = reading
		time = f.read()
		f.close 
		f = open("final_score.txt")
		final_score = f.read()
		f.close
		f = open("max_angle.txt")
		max_angle = f.read()
		f.close
		f = open('demofile.txt') #r = reading
		count = f.read()
		f.close 
		#f = open("e.txt")
		#e = f.read()
		#f.close
		#f = open("f.txt")
		#f = f.read()
		#f.close
		#return JsonResponse({'a':a, 'b':b, 'c':c, 'max_angle':max_angle,'e':e,'f':f})
		return JsonResponse({'time':time, 'final_score': final_score, 'max_angle':max_angle, 'count':count})
		#return JsonResponse({'count':counter})

	#else:
	print("**********************")
	print("Hello")
	return render(request, 'blog/about.html')
    

def abc(request):
	return render(request, 'blog/home.html')
