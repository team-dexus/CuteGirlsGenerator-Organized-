//created by JPNYKW

var wid=128,hig=128; // for draw engine

CanvasRenderingContext2D.prototype.drawImageByData=function($Array,$width,$height,$find,$px,$x,$y){
	let $dy=$y+$px/2;
	let $idx=$find;
	for($hc=0;$hc<$height;$hc++){
		let $dx=$x+$px/2;
		for($wc=0;$wc<$width;$wc++){
			let R=$Array[$idx + ($Array.length / 3) * 0]*127.5+127.5;
			let G=$Array[$idx + ($Array.length / 3) * 1]*127.5+127.5;
			let B=$Array[$idx + ($Array.length / 3) * 2]*127.5+127.5;
			R=~~R+',';G=~~G+',';B=~~B+'';
			this.drawFillBox($dx,$dy,$px+1,`rgb(${R+G+B})`);
			$dx+=$px;
			$idx+=1;
		}
		$dy+=$px;
	}
	// debugger fpr console
	c.timeEnd();
	c.log('%cDebug','color:#0093EF;font-weight:bold',`id:${$idx}`);
	c.log('%cDebug %cSuccessful','color:#0093EF;font-weight:bold','color:#F53300;font-weight:bold','draw an Image');
}

function drawGeneratedImage(){
      cont.clearRect(0,0,canv.width,canv.height);
      cont.drawImageByData(generatedData,wid,hig,0,2,0,0);
}
