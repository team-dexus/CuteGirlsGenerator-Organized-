var download, onGenerated = false;
var fileIndex = 0;
var d = document;
var c = console;
var w = window;
var m = Math;
var w, h;
var canv, cont;
var generatedData; // generate2() の戻り値を格納する変数
window.onload = () => {
	download = d.getElementById("download");
	canv = d.getElementById('output');
	cont = canv.getContext('2d');
	w = d.getElementById('w');
	h = d.getElementById('h');
	canv.width = 256;
	canv.height = 256;
	textRoll('-Generate ultimate cute girls-', d.getElementById('display'), 50, d.getElementById('shel'), '<h1 class="logo">-Generate <span class="text">ultimate cute</span> girls.-</h1>');
	tag=str=>str.split(/[\　\ ]+/ig);

	//モデルの読み込み
	d.getElementById("generate").setAttribute("disabled", true)
	d.getElementById("download").setAttribute("disabled", true)
	d.getElementById("genOrLoad").textContent = "loading..."
	WebDNN.load('./output')
   .then(function(loaded){
       console.log('loaded');
			 runner = loaded
			 console.log(runner.backendName);
			 d.getElementById("generate").removeAttribute("disabled")
			 d.getElementById("download").removeAttribute("disabled")
			 d.getElementById("genOrLoad").textContent = "Generate!"
       // add your code here.
   });
}

function textRoll(text, target, interval, subtarget, afterInput) {
	let cnt = 0;
	let roll = setInterval(() => {
		cnt++;
		target.innerText = text.substr(0, cnt);
		if (cnt > text.length) {
			clearInterval(roll);
			if (afterInput != void(0)) subtarget.innerHTML = afterInput;
		}
	}, interval);
}

function setImgSize() {
	wid = ~~w.value;
	hig = ~~h.value;
	console.log([w, h]); // test
}

function downloadHandle() {
	var DLImg = document.createElement("a")
	DLImg.download = `CGG${++fileIndex}.png`
	DLImg.href = canv.toDataURL()
	DLImg.click()
}
