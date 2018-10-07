function generate() {
	let pic = document.getElementById('GirlPic'),
	item = ['MG7.png', 'MG57.png', 'code100.png'],
	text = document.getElementById('words').value,
	arrayRes = document.getElementById('ArrayResult');
	//result = '===WORDS===<br>';

	tag(text).forEach((e, i) => result += `<span id=res${i}>${e}</span><br>`);
	document.getElementById('ShowText').textContent = `Generation result in ${text}`;
	//document.getElementById('ArrayRes').innerHTML = `${result}=========`;
	pic.src = item[~~(Math.random() * 3)];
}

function random_norm() {
  //平均0、標準偏差1の正規乱数を返す。
  var s, i;
  s = 0;
  for(i = 0; i < 12; i++) {
    s += Math.random();
  }
  return s - 6;
}

function generate2() {
	document.getElementById('generate').setAttribute('disabled', true);
	onGenerated && document.getElementById('download').setAttribute('disabled', true);
	onGenerated = false;

	// Generate
	let x = runner.inputs[0], y = runner.outputs[0], noise;
	threshold = 0.3
	noise_size = 128;
	noise = new Float32Array(noise_size);
	for(let i = 0; i < noise_size; i++) {
		do {
			noise[i] = random_norm()
		}while(Math.abs(noise[i]) > threshold);
	}

/*
	numberOfTag = 1539;
	var tags = new Float32Array(numberOfTag);
	for(let i = 0; i < 4; i++){
		tags[Math.floor(Math.random() * 1539) - 1] = 1;
	}
/*
	for(let i = 0; i < 1539; i++){
		tags[i] = Math.random();
	}
*/

	// Select the tag
	x.set(noise);
	console.log(x);

	runner.run().then(function() {
		console.log('Before', y); // debug
		let y_typed_array = y.toActual();
		console.log('After', y_typed_array);

		if (y_typed_array.includes(NaN)) {
			console.timeEnd();
			alert('Failed the generated.');
		} else {
			console.log('finished');
			generatedData = y_typed_array;
			drawGeneratedImage(generatedData);
		}

		onGenerated = true;
		document.getElementById('generate').removeAttribute('disabled');
		document.getElementById('download').removeAttribute('disabled');
    });
}
