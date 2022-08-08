프레임 간의 관계Window.open() 메서드는 원본 창 객체를 가리키는 opener 프로퍼티를 가진 새로운 Window 객체를 반환한다. 
이런식으로 서로를 참조할 수 있고, 서로 간의 프로퍼티를 읽고 상대방의 메서드를 호출할 수도 있다. 비슷한 일이 프레임 사이에서도 가능하다. 
창이나 프레임 내에서 실행되는 코드는 프레임을 포함하고 있는 창, 혹은 프레임이나 중첩된 자식 프레임을 다음 프로퍼티들을 사용해서 참조할 수 있다.

parent.history.back();/**프레임에서 다른 창이나 프레임의 Window 객체를 참조하려면 parent프로퍼티를 사용하면 된다.
*/창이나 프레임의 하위 프레임을 참조할 수 있는 방법도 있다. 문서 내에 <iframe id="f1">요소가 있을 때,
// iframe의 Element객체를 가져옴var elt = document.getElementById("f1");// iframe요소에는 해당 프레임의 window객체를 가리키는 contentWindow프로퍼티가 존재한다.
var win = elt.contentWindow;// win이 프레임의 Window객체이므로 참이다.win.frameElement === elt;    
// window는 최상위 창의 Window 객체이므로 null이다.window.frameElement === null;모든 Window객체에는 해당 창이나 프레임 내에 
포함된 자식 프레임들의 목록인 frames 프로퍼티가 존재한다. frames 프로퍼티는 인덱스 번호나 프레임의 이름으로 참조 가능한 유사 배열이다. 

frames[] 유사 배열의 각 요소는 <iframe>요소가 아니라 Window 객체이다.* 창의 첫 번째 자식 프레임을 참조하려면 frames[0]을 사용하면 된다. 
* 두번째 자식 프레임 내부에 존재하는 세번째 자식 프레임을 참조하려면, frames[1].frames[2]를 사용하면 된다. 
* 프레임 내에서 실행되는 코드에서 이웃하고 있는 프레임을 참조하기 위해 parent.frames[1]과 같은 코드를 사용할 수도 있다.
* <iframe>요소에 id나 name속성을 지정했다면, frames["f1"] 또는 frames.f1을 사용해서 참조할 수 있다.
* 프레임들의 갯수를 알아내기 위해 window.length를 사용할 수 있다.창들과 상호작용하는 자바스크립트각 창이나 프레임에는 
고유의 자바스크립트 실행 문맥과 전역 객체(Window)가 존재한다. 그런데, 코드에서 어떤 창이나 프레임이 다른 창이나 프레임을 참조할 수 있다면
(그리고 동일 출처 정책을 만족한다면), 이들과 상호작용할 수 있다.// 웹 페이지에 A와 B라는 이름의 두 <iframe>요소가 존재한다고 가정
// 프레임 A에서 변수 i를 선언.var i = 3;// 프레임 B에서 프레임 A의 변수 i에 접근해서 변수의 값을 변경parent.A.i =4; 
// 프레임 B에서 f()함수를 선언.function f() { };// 프레임 A에서 프레임 B의 f()함수를 호출parent.B.f();
