const defaultSize = 256;

(() => {
      window.onload = init;

      function init() {
            generatedData = null;
            onGenerated = false;
            fileIndex = 0;

            download = document.getElementById('download');
            canv = document.getElementById('output');
            cont = canv.getContext('2d');

            w = document.getElementById('w').value;
            h = document.getElementById('h').value;
            w = w === '' ? defaultSize : ~~w;
            h = h === '' ? defaultSize : ~~h;

            canv.width = canv.height = 335;

            // Onload data of models.
            let button = {
                  generate: document.getElementById('generate'),
                  download: document.getElementById('download')
            };

            Object.keys(button).map(key => {
                  button[key].setAttribute('disabled', true);
                  button[key].style.background = '#BEBEBE';
            });

            WebDNN.load('./output').then(loaded => {
            	console.log('loaded');
            	console.log(runner.backendName);

                  Object.keys(button).map(key => {
                        button[key].removeAttribute('disabled');
                        button[key].style.background = '#15D1F9';
                  });

                  runner = loaded;
            });

            function setImgSize() {
            	wid = ~~w.value;
            	hig = ~~h.value;
            	console.log(w, h);
            }

            function downloadHandle() {
            	let  downloadDom = document.createElement('a');
            	downloadDom.download = `CGG${++fileIndex}.png`;
            	downloadDom.href = canv.toDataURL();
            	downloadDom.click();
            }
      }
})();
