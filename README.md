# imgaug
Please checkout the original repository for the amazing work by ***aleju*** and more detailed introduction.  
(https://github.com/aleju/imgaug)  

## What's new
Two more types of augmentation are added in the repository, **Spot** and **Flare**.  
### Spot
![spot](https://wx2.sinaimg.cn/mw690/8a44a48egy1fprfws33p3j20qo0hvwfn.jpg)  
The two paraters of *sigma* indicates the uppper and lower boundaries of the radius of the spots.
### Flare
![flare](https://wx3.sinaimg.cn/mw690/8a44a48egy1fprfwozykpj20qo0hvjtu.jpg)  

## Installation
**This is just one way of importing, don't have to follow the exact steps.**  

Suppose you have a project called *YourProject* in need for data augmentation.  
1. Clone this repository into *YourProject*  
2. Write yourself a test script under the , e.g. *test.py*  
Your directory should be like this:  
*YourProject*
    --imgaug  
    --*test.py*  
3. In *test.py*, write:  
`
from imgaug.imgaug import augmenters as iaa
`   
