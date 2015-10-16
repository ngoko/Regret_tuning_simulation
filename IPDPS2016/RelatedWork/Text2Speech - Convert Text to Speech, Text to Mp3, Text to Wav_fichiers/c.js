(function(d){d.fn.textareaCount=function(b,q){function f(){g.html(t());"undefined"!=typeof q&&q.call(this,{input:e,max:h,left:m,words:k});return!0}function t(){var a=c.val(),n=a.length;if(0<b.maxCharacterSize){n>=b.maxCharacterSize&&(a=a.substring(0,b.maxCharacterSize));var l=r(a),d=b.maxCharacterSize-l;p()||(d=b.maxCharacterSize);if(n>d){var f=this.scrollTop;c.val(a.substring(0,d));this.scrollTop=f}g.removeClass(b.warningStyle);d-n<=b.warningNumber&&g.addClass(b.warningStyle);e=c.val().length+l;p()||(e=c.val().length);k=s(c.val()).length-1;m=h-e}else l=r(a),e=c.val().length+l,p()||(e=c.val().length),k=s(c.val()).length-1;a=b.displayFormat;a=a.replace("#input",e);a=a.replace("#words",k);0<h&&(a=a.replace("#max",h),a=a.replace("#left",m));return a}function p(){return-1!=navigator.appVersion.toLowerCase().indexOf("win")?!0:!1}function r(a){for(var b=0,c=0;c<a.length;c++)"\n"==a.charAt(c)&&b++;return b}function s(a){a=(a+" ").replace(/^[^A-Za-z0-9]+/gi,"");var b=rExp=/[^A-Za-z0-9]+/gi;return a.replace(b," ").split(" ")}b=d.extend({maxCharacterSize:-1,originalStyle:"originalTextareaInfo",warningStyle:"warningTextareaInfo",warningNumber:20,displayFormat:"#input characters | #words words"},b);var c=d(this);d("<div class='charleft text-right'>&nbsp;</div>").insertAfter(c);var g=c.next(".charleft");g.addClass(b.originalStyle);g.css({});var e=0,h=b.maxCharacterSize,m=0,k=0;c.bind("keyup",function(a){f()}).bind("mouseover",function(a){setTimeout(function(){f()},10)}).bind("paste",function(a){setTimeout(function(){f()},10)})}})(jQuery); 
(function(e){var t,o={className:"autosizejs",id:"autosizejs",append:"\n",callback:!1,resizeDelay:10,placeholder:!0},i='<textarea tabindex="-1" style="position:absolute; top:-999px; left:0; right:auto; bottom:auto; border:0; padding: 0; -moz-box-sizing:content-box; -webkit-box-sizing:content-box; box-sizing:content-box; word-wrap:break-word; height:0 !important; min-height:0 !important; overflow:hidden; transition:none; -webkit-transition:none; -moz-transition:none;"/>',n=["fontFamily","fontSize","fontWeight","fontStyle","letterSpacing","textTransform","wordSpacing","textIndent"],s=e(i).data("autosize",!0)[0];s.style.lineHeight="99px","99px"===e(s).css("lineHeight")&&n.push("lineHeight"),s.style.lineHeight="",e.fn.autosize=function(i){return this.length?(i=e.extend({},o,i||{}),s.parentNode!==document.body&&e(document.body).append(s),this.each(function(){function o(){var t,o=window.getComputedStyle?window.getComputedStyle(u,null):!1;o?(t=u.getBoundingClientRect().width,(0===t||"number"!=typeof t)&&(t=parseInt(o.width,10)),e.each(["paddingLeft","paddingRight","borderLeftWidth","borderRightWidth"],function(e,i){t-=parseInt(o[i],10)})):t=p.width(),s.style.width=Math.max(t,0)+"px"}function a(){var a={};if(t=u,s.className=i.className,s.id=i.id,d=parseInt(p.css("maxHeight"),10),e.each(n,function(e,t){a[t]=p.css(t)}),e(s).css(a).attr("wrap",p.attr("wrap")),o(),window.chrome){var r=u.style.width;u.style.width="0px",u.offsetWidth,u.style.width=r}}function r(){var e,n;t!==u?a():o(),s.value=!u.value&&i.placeholder?(p.attr("placeholder")||"")+i.append:u.value+i.append,s.style.overflowY=u.style.overflowY,n=parseInt(u.style.height,10),s.scrollTop=0,s.scrollTop=9e4,e=s.scrollTop,d&&e>d?(u.style.overflowY="scroll",e=d):(u.style.overflowY="hidden",c>e&&(e=c)),e+=w,n!==e&&(u.style.height=e+"px",f&&i.callback.call(u,u))}function l(){clearTimeout(h),h=setTimeout(function(){var e=p.width();e!==g&&(g=e,r())},parseInt(i.resizeDelay,10))}var d,c,h,u=this,p=e(u),w=0,f=e.isFunction(i.callback),z={height:u.style.height,overflow:u.style.overflow,overflowY:u.style.overflowY,wordWrap:u.style.wordWrap,resize:u.style.resize},g=p.width(),y=p.css("resize");p.data("autosize")||(p.data("autosize",!0),("border-box"===p.css("box-sizing")||"border-box"===p.css("-moz-box-sizing")||"border-box"===p.css("-webkit-box-sizing"))&&(w=p.outerHeight()-p.height()),c=Math.max(parseInt(p.css("minHeight"),10)-w||0,p.height()),p.css({overflow:"hidden",overflowY:"hidden",wordWrap:"break-word"}),"vertical"===y?p.css("resize","none"):"both"===y&&p.css("resize","horizontal"),"onpropertychange"in u?"oninput"in u?p.on("input.autosize keyup.autosize",r):p.on("propertychange.autosize",function(){"value"===event.propertyName&&r()}):p.on("input.autosize",r),i.resizeDelay!==!1&&e(window).on("resize.autosize",l),p.on("autosize.resize",r),p.on("autosize.resizeIncludeStyle",function(){t=null,r()}),p.on("autosize.destroy",function(){t=null,clearTimeout(h),e(window).off("resize",l),p.off("autosize").off(".autosize").css(z).removeData("autosize")}),r())})):this}})(window.jQuery||window.$);
$.fn.upload=function(h,b,d,a){"object"!=typeof b&&(a=d,d=b);return this.each(function(){if($(this)[0].files[0]){var e=new FormData;e.append($(this).attr("name"),$(this)[0].files[0]);if("object"==typeof b)for(var g in b)e.append(g,b[g]);$.ajax({url:h,type:"POST",xhr:function(){myXhr=$.ajaxSettings.xhr();myXhr.upload&&a&&myXhr.upload.addEventListener("progress",function(f){var c=~~(f.loaded/f.total*100);a&&"function"==typeof a?a(f,c):a&&$(a).val(c)},!1);return myXhr},data:e,dataType:"json",cache:!1,
contentType:!1,processData:!1,complete:function(a){var c;try{c=JSON.parse(a.responseText)}catch(b){c=a.responseText}d&&d(c)}})}})};
function analytics($url){
  //history.pushState && history.replaceState && history.pushState({id: 1}, document.title, $url);
         _gaq.push(["_trackPageview", $url/*window.location.pathname*/]);
};
$(document).ready(function(){function b(){$("#archivo").upload("subir_archivo.php",{nombre_archivo:"temp"},function(a){$("#btn-upload").html('<span class="glyphicon glyphicon-file"></span>  Upload text file');$("#t").val(a).trigger("autosize.resize")},function(a,b){$("#btn-upload").html('<span class="glyphicon glyphicon-time"></span> '+b+" %")})}$(".textarea").autosize();$("#textoid").autosize();$("#t").textareaCount({maxCharacterSize:2000,originalStyle:"originalTextareaInfo",warningStyle:"warningTextareaInfo",
warningNumber:40,displayFormat:"#input/#max"});$(".from").click(function(){$("#lt").val()!=$(this).data("code")&&($code=$(this).data("code"),$idioma=$(this).data("name"),$imagen='<img src="flag/'+$code+'.png" width="22" height="15" alt="'+$idioma+'">',$("#dropdownlangfrom").html($imagen+" "+$idioma+" <span class=caret></span>"),$("#lf").val($code))});$("#trash").click(function(){$("#t").val("");$("#t").height(0)});$("#mp3from").click(function(){analytics("/mp3");text=$("#t").val();text.replace("\n"," ");lang=$("#lf").val();
url=encodeURI("http://text2speech.us/mp3.php?t="+text+"&lf="+lang);""!=text?document.location=url:alert("Please enter some text.")});$("#wavfrom").click(function(){analytics("/wav");text=$("#t").val();text.replace("\n"," ");lang=$("#lf").val();url=encodeURI("http://text2speech.us/wav.php?t="+text+"&lf="+lang);""!=text?document.location=url:alert("Please enter some text.")});$("#playMp3from").click(function(){text=$("#t").val();text.replace("\n"," ");lang=$("#lf").val();mp3=encodeURI("http://text2speech.us/mp3.php?t="+text+"&lf="+
lang);wav=encodeURI("http://text2speech.us/wav.php?t="+text+"&lf="+lang);var a=document.getElementById("player");document.getElementById("sourceWav");a.paused&&""!=text?($("#lo").val("true"),-1!=navigator.userAgent.indexOf("MSIE")?a.src=mp3:-1!=navigator.userAgent.indexOf("Trident")?a.src=mp3:-1!=navigator.userAgent.indexOf("OPR")?a.src=wav:-1!=navigator.userAgent.indexOf("Opera")?a.src=wav:-1!=navigator.userAgent.indexOf("Firefox")?a.src=wav:-1!=navigator.userAgent.indexOf("Chrome")?a.src=mp3:-1!=navigator.userAgent.indexOf("Safari")?
a.src=mp3:a.src=wav,a.play()):a.pause()});player.addEventListener("loadstart",function(){"true"==$("#lo").val()&&($("#playMp3from").html('<span class="glyphicon glyphicon-time"></span> Loading'),$("#playMp3from").toggleClass("loading"),$("#t").attr("readonly",!0))},!1);player.addEventListener("loadeddata",function(){$("#playMp3from").html('<span class="glyphicon glyphicon-stop"></span> Stop');$("#playMp3from").toggleClass("loading");$("#t").attr("readonly",!1)},!1);player.addEventListener("play",
function(){analytics("/play"); $("#playMp3from").html('<span class="glyphicon glyphicon-stop"></span> Stop');$("#t").attr("readonly",!1)},!1);player.addEventListener("pause",function(){$("#playMp3from").html('<span class="glyphicon glyphicon-play"></span> Listen')},!1);$("#btn-upload").click(function(a){$("#archivo").click()});document.getElementById("archivo").onchange=function(){$("#boton_subir").click()};$("#boton_subir").on("click",function(){b()})});