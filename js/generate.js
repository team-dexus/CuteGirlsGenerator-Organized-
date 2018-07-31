
function generate(){ //Added by mirukuma ,Changed by JPNYKW ,Changed by mirukuma ,Changed by JPNYKW
  var pic=d.getElementById('GirlPic');
  var item=['MG7.png','MG57.png','code100.png']
  var text=d.getElementById('words').value;
  var arrayRes=d.getElementById('ArrayResult');
  var result='===WORDS===<br>';
  tag(text).forEach(function(e,i,a){result+=`<span id=res${i}>${e}</span><br>`});
  d.getElementById('ShowText').textContent='Generation result in '+text;
  d.getElementById('ArrayRes').innerHTML=`${result}=========`;
  pic.src=item[~~(Math.random()*3)];


}

function generate2(){
  d.getElementById("generate").setAttribute("disabled", true)
  if (onGenerated) d.getElementById("download").setAttribute("disabled", true)
  onGenerated=false; //Added by JPNYKW

  //生成
  let x = runner.inputs[0];
  let y = runner.outputs[0];

  noise_size = 128;
  var noise= new Float32Array(noise_size)

  for(i=0; i<noise_size; i=i+1){
    noise[i]=Math.random()*2-1
  }

  /*
  numberOfTag=1539

  var tags= new Float32Array(numberOfTag)


  for(i=0; i<4; i=i+1){
    tags[Math.floor(Math.random()*1539)-1]=1
  }
  /*
  for(i=0; i<1539; i=i+1){
    tags[i]=Math.random()
  }
  */

  //タグの指定

  x.set(noise);
  console.log(x)
  console.log("sex")
  runner.run()
    .then(function() {
      console.log('finished');
      let y_typed_array = y.toActual();
      console.log(y_typed_array);
      generatedData=y_typed_array; //Added by JPNYKW
      drawGeneratedImage(); //Added by JPNYKW
      onGenerated=true; //Added by JPNYKW
      d.getElementById("generate").removeAttribute("disabled")
      d.getElementById("download").removeAttribute("disabled")

    });
}
