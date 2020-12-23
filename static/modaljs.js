document.getElementbyId('button1').addEventListener('click',function(){
	document.querySelector('.bg-modal').style.display = 'flex';
});

document.querySelector('.cross').addEventListener('click',function(){
	document.querySelector('.bg-modal').style.display = 'none';
});
