var onGenerated = false, fileIndex = 0, generatedData; // stack value of return by generate2()
tag = str => str.split(' '); // split strings of tag

window.onload = function() {
	download = document.getElementById('download');
	canv = document.getElementById('output');
	cont = canv.getContext('2d');
	w = document.getElementById('w');
	h = document.getElementById('h');
	canv.width = 256;
	canv.height = 256;
	
	textRoll('-Generate ultimate cute girls-', document.getElementById('display'), 50, document.getElementById('shel'), '<h1 class="logo">-Generate <span class="text">ultimate cute</span> girls.-</h1>');

	// Onload data of models.
	document.getElementById('generate').setAttribute('disabled', true);
	document.getElementById('download').setAttribute('disabled', true);
	document.getElementById('genOrLoad').textContent = 'loading...';
	WebDNN.load('./output').then(function(loaded){
		console.log('loaded');
		runner = loaded;
		console.log(runner.backendName);
		document.getElementById('generate').removeAttribute('disabled');
		document.getElementById('download').removeAttribute('disabled');
		document.getElementById('genOrLoad').textContent = 'Generate!';
       // add your code here.
   });
}

function textRoll(text, target, interval, subtarget, afterInput) {
	let cnt = 0;
	let roll = setInterval(function() {
		cnt++;
		target.innerText = text.substr(0, cnt);
		if(cnt > text.length) {
			clearInterval(roll);
			if (afterInput) subtarget.innerHTML = afterInput;
		}
	}, interval);
}

function setImgSize() {
	wid = ~~w.value;
	hig = ~~h.value;
	console.log([w, h]); // debug
}

function downloadHandle() {
	var DLImg = document.createElement('a');
	DLImg.download = `CGG${++fileIndex}.png`;
	DLImg.href = canv.toDataURL();
	DLImg.click();
}
