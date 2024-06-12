# diffusion-model
本项目是使用diffusion model生成数字手写图像。
首先运行diffusion.py对mnist数据集训练，一共训练10epoch，保存最后一次生成的模型。
再运行mian.py，调用load.py加载模型；在main中，使用控制台输入streamlit run main.py来运行项目，控制台出现以下画面，并自动跳转浏览器打开网页。
![image](https://github.com/2820214325/diffusion-model/assets/88656477/0cc14ce6-23cc-42ef-bc0b-6203df50394b)
网页如图所示：
![image](https://github.com/2820214325/diffusion-model/assets/88656477/6edef99c-5db4-430a-9e47-57e5dcf15eaf)
在网页中，可以输入想要生成的数字，也可以选择随机生成数字，但由于本项目是加载训练过后的模型，所以没办法自己选择生成步数。
