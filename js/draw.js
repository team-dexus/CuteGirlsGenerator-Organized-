wid = 128, hig = 256; // for draw engine
CanvasRenderingContext2D.prototype.drawImageByData = function(ary, width, height, find, px, x, y) {
	let z = ary.length / 3,
		dy = y + px / 2,
		idx = find;

	for(let hc = 0; hc < height; hc++) {
		let dx = x + px / 2;
		for(let wc = 0; wc < width; wc++) {
			let R = ary[idx + z * 0] * 127.5 + 127.5;
			let G = ary[idx + z * 1] * 127.5 + 127.5;
			let B = ary[idx + z * 2] * 127.5 + 127.5;
			this.drawFillBox(dx, dy, px + 1,`rgb(${~~R},${~~G},${~~B})`);
			dx += px;
			idx++;
		}
		dy += px;
	}
	// debug
	console.timeEnd();
	console.log('%cDebug', 'color:#0093EF;font-weight:bold', `id:{idx}`);
	console.log('%cDebug %cSuccessful', 'color:#0093EF;font-weight:bold', 'color:#F53300;font-weight:bold', 'draw an Image');
}

function drawGeneratedImage(data) {
      cont.clearRect(0, 0, canv.width, canv.height);
      cont.drawImageByData(data, wid, hig, 0, 1, 0, 0);
}
