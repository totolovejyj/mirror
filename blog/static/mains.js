$(window).ready(function(){
  drawpie(50, '.pie-chart1', '#ccc');
  drawpie(50, '.pie-chart2', '#8b22ff');
  drawpie(50, '.pie-chart3','#ff0');
});

function drawpie(max, classname, colorname){
   console.log(arguments[0])
   var i=1;
    var func1 = setInterval(function(){
      if(i<max){
          color1(i,classname,colorname);
          i++;
      } else{
        clearInterval(func1);
      }
    },10);
}

function color1(i, classname,colorname){
  $(classname).css({
       "background":"conic-gradient("+colorname+" 0% "+i+"%, #ffffff "+i+"% 100%)"
   });
}

