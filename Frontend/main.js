// ==========================
// main.js  (FULL REPLACEMENT)
// ==========================

// API 베이스 (config.js가 먼저 로드되면 window.API_BASE 사용)
const BASE = (window.API_BASE || "").replace(/\/+$/, "") || "";

// --------------------------
// 페이지 로드 애니메이션 등
// --------------------------
jQuery(window).on('load', function() {
  "use strict";

  // HIDE PRELOADER
  $(".preloader").addClass("hide-preloader");

  // SHOW/ANIMATE ANIMATION CONTAINER
  setTimeout(function(){
    $("#intro .animation-container").each(function() {
      var e = $(this);
      setTimeout(function(){
        e.addClass("run-animation");
      }, e.data("animation-delay") );
    });
  }, 800);
});

jQuery(document).ready(function($) {
  "use strict";

  // ONE PAGE NAVIGATION
  $(".navigation-main .navigation-items").onePageNav({
    currentClass: "current",
    changeHash: false,
    scrollSpeed: 750,
    scrollThreshold: 0.5,
    filter: ":not(.external)",
    easing: "swing"
  });

  // SMOOTH SCROLL FOR SAME PAGE LINKS
  $(document).on('click', 'a.smooth-scroll', function(event) {
    event.preventDefault();
    $('html, body').animate({
      scrollTop: $( $.attr(this, 'href') ).offset().top
    }, 500);
  });

  // INIT PARALLAX PLUGIN
  $(".background-content.parallax-on").parallax({
    scalarX: 24,
    scalarY: 15,
    frictionX: 0.1,
    frictionY: 0.1,
  });

  // SCROLL REVEAL SETUP
  window.sr = ScrollReveal();
  sr.reveal(".scroll-animated-from-bottom", {
    duration: 600,
    delay: 0,
    origin: "bottom",
    rotate: { x: 0, y: 0, z: 0 },
    opacity: 0,
    distance: "20vh",
    viewFactor: 0.4,
    scale: 1,
  });

  // WORK CAROUSEL
  $('.work-carousel').owlCarousel({
    center: true,
    items: 1,
    loop: true,
    margin: 30,
    autoplay: true,
    responsive:{
      800:{ items: 3 },
    }
  });


  // ------------------------------------------
  // === Service Upload UI (UPLOAD→JOB→POLL) ===
  // ------------------------------------------
  const $form    = $('#service-upload');
  if(!$form.length) return;

  const $dropImg = $('#drop-zone');          // 이미지 드롭존
  const $imgInput= $('#file-input');         // 이미지 input
  const $preview = $('#upload-preview');
  const $prompt  = $dropImg.find('.upload-prompt');
  const $status  = $('#upload-status');
  const $message = $('#upload-message');

  // 오디오/보이스 옵션
  const $audioInput = $('#audio-input');     // <input type="file" accept="audio/wav" required>
  const $audioDrop  = $('#audio-drop');      // 오디오 드롭존
  const $audioName  = $('#audio-filename');
  const $audioPrev  = $('#audio-preview');

  const $selGender  = $('#voice-gender');    // male|female
  const $selAge     = $('#voice-age');       // young|middle|old
  const $rangePitch = $('#voice-level');     // -20 ~ 20

  let selectedImage = null;  // 이미지 파일
  let selectedAudio = null;  // 오디오 파일 (DnD/클릭 공용)

  // ---------- 프리뷰/UX ----------
  function resetImgPreview(){
    $preview.hide().attr('src','');
    $prompt.show();

    // ★ 추가: 결과 비디오도 숨기고 src 제거
    var $vid = $('#result-video');
    if ($vid.length) $vid.hide().attr('src','');
  }
  function showImgPreview(file){
    if(!file){ resetImgPreview(); return; }
    if(file.type && file.type.indexOf('image') === 0){
      const url = URL.createObjectURL(file);
      $preview.attr('src', url).show();
      $prompt.hide();
    } else {
      resetImgPreview();
    }
  }
  function showAudioPreview(file){
    if(!file) return;
    if ($audioName) $audioName.text(file.name);
    if ($audioPrev && $audioPrev.length){
      $audioPrev[0].src = URL.createObjectURL(file);
      $audioPrev.show();
      $audioPrev[0].onloadeddata = () => URL.revokeObjectURL($audioPrev[0].src);
    }
  }

  // ---------- 이미지 드래그/선택 ----------
  $dropImg.on('click', () => $imgInput.trigger('click'));
  $dropImg.on('keydown', (e) => {
    if(e.key === 'Enter' || e.key === ' '){
      e.preventDefault(); $imgInput.trigger('click');
    }
  });
  $dropImg.on('drag dragstart dragend dragover dragenter dragleave drop', function(e){
    e.preventDefault(); e.stopPropagation();
  }).on('dragover dragenter', () => $dropImg.addClass('dragover')
  ).on('dragleave dragend drop', () => $dropImg.removeClass('dragover')
  ).on('drop', function(e){
    const dt = e.originalEvent.dataTransfer;
    if(dt && dt.files && dt.files.length){
      selectedImage = dt.files[0];
      // input.files 동기화 (가능한 브라우저에서)
      try{
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(selectedImage);
        $imgInput[0].files = dataTransfer.files;
      }catch(_){}
      showImgPreview(selectedImage);
    }
  });

  $imgInput.on('change', function(){
    selectedImage = this.files && this.files[0] ? this.files[0] : null;
    showImgPreview(selectedImage);
  });

  // ---------- 오디오 드래그/선택 ----------
  if ($audioDrop && $audioDrop.length){
    ['dragenter','dragover'].forEach(ev =>
      $audioDrop.on(ev, e => { e.preventDefault(); e.stopPropagation(); $audioDrop.addClass('dragover'); })
    );
    ['dragleave','drop'].forEach(ev =>
      $audioDrop.on(ev, e => { e.preventDefault(); e.stopPropagation(); $audioDrop.removeClass('dragover'); })
    );
    $audioDrop.on('drop', e => {
      const files = e.originalEvent.dataTransfer?.files;
      if (files && files.length) {
        const f = files[0];
        if (f.type === 'audio/wav' || /\.wav$/i.test(f.name)) {
          selectedAudio = f;
          // input.files 동기화 (required 통과용)
          try {
            const dt = new DataTransfer();
            dt.items.add(f);
            $audioInput[0].files = dt.files;
          } catch (err) {
            console.warn('DataTransfer 주입 실패, selectedAudio로 진행:', err);
          }
          showAudioPreview(f);
        } else {
          alert('WAV 파일만 업로드할 수 있습니다.');
        }
      }
    });
  }

  $audioInput.on('change', function(){
    const f = this.files && this.files[0] ? this.files[0] : null;
    if (f){
      selectedAudio = f;
      showAudioPreview(f);
    }
  });

  // ---------- 보이스 옵션 ----------
  function getVoiceProfile(){
    const gender = ($selGender && $selGender.val()) || 'female'; // male|female
    let   age    = ($selAge && $selAge.val())       || 'young';  // young|middle|old
    if (age === 'middle' || age === 'old') age = 'adult';        // 서버 프로필과 매핑
    return `${gender}_${age}`; // 예: female_young, male_adult
  }
  function getPitch(){
    const raw = ($rangePitch && $rangePitch.val()) || '0';
    const n = Number(raw);
    return Number.isFinite(n) ? n : 0;
  }

  // ---------- API 유틸 ----------
  async function uploadToServer(imgFile, msg, audioFile){
    const fd = new FormData();
    fd.append('file', imgFile, imgFile.name || 'upload.png');
    if (audioFile) fd.append('audio', audioFile, audioFile.name || 'voice.wav');
    if (msg) fd.append('message', msg);

    const res = await fetch(`${BASE}/api/upload`, { method:'POST', body: fd });
    if(!res.ok){
      const t = await res.text();
      throw new Error(`/api/upload 실패: ${res.status} ${t.slice(0,200)}`);
    }
    const json = await res.json();
    if(!json || !json.image_path){
      throw new Error('업로드 응답에 image_path가 없습니다.');
    }
    return json; // { ok, image_path, audio_path|null, message }
  }

  async function createJob(image_path, audio_path){
    if (!audio_path) throw new Error('오디오가 업로드되지 않았습니다(audio_path 없음). WAV를 선택하세요.');

    const body = {
      image_path,
      audio_path,              // 백엔드 필수
      use_applio: true,
      pitch: getPitch(),
      voice_profile: getVoiceProfile()
    };

    const res = await fetch(`${BASE}/jobs/audio`, {
      method: 'POST',
      headers: { 'Content-Type':'application/json' },
      body: JSON.stringify(body)
    });
    if(!res.ok){
      const t = await res.text();
      throw new Error(`/jobs/audio 실패: ${res.status} ${t.slice(0,200)}`);
    }
    return res.json(); // { job_id, status }
  }

  async function pollJob(job_id, onProgress){
    while(true){
      const res = await fetch(`${BASE}/jobs/${job_id}`);
      if(!res.ok){
        const t = await res.text();
        throw new Error(`/jobs/${job_id} 실패: ${res.status} ${t.slice(0,200)}`);
      }
      const j = await res.json();
      if (onProgress && j.step) onProgress(j.step);

      if(j.status === 'done')  return j;
      if(j.status === 'failed') throw new Error(`잡 실패: ${j.error || '알 수 없는 오류'}`);

      await new Promise(r => setTimeout(r, 1500));
    }
  }

  // 결과 URL 빌더: 절대/상대 경로 모두 대응
  function buildFileUrl(p){
    if (!p) return '';
    if (p.startsWith('/')) {
      return `${BASE}/files?path=${encodeURIComponent(p)}`;   // 절대경로
    }
    return `${BASE}/files/${encodeURIComponent(p)}`;          // 상대경로
  }

  function setBusy(b){
    $form.find('button, input, select, textarea').prop('disabled', b);
  }

  // ---------- 제출 ----------
  $form.on('submit', async function(e){
    e.preventDefault();

    // 이미지 필수
    if(!selectedImage){
      $status.text('이미지 파일을 선택하세요.');
      return;
    }

    // 오디오 필수
    const audioFile = selectedAudio
      || ($audioInput && $audioInput[0] && $audioInput[0].files && $audioInput[0].files[0])
      ? (selectedAudio || $audioInput[0].files[0])
      : null;
    if (!audioFile) {
      $status.text('WAV 오디오 파일을 선택하세요.');
      return;
    }

    try{
      setBusy(true);
      $status.text('업로드 중...');

      // 1) 업로드 → 경로 획득
      const up = await uploadToServer(selectedImage, $message.val(), audioFile);

      if (!up.audio_path) {
        $status.text('오디오 업로드 실패: audio_path가 없습니다. WAV 파일을 다시 선택해 주세요.');
        return;
      }

      // 2) 잡 생성
      $status.text('잡 생성 중...');
      const job = await createJob(up.image_path, up.audio_path);

      // 3) 폴링
      $status.text('처리 중입니다... (applio → sadtalker → wav2lip → gfpgan)');
      const result = await pollJob(job.job_id, step => $status.text(`진행 중: ${step}...`));

      // 4) 결과 재생
      const finalRel = result?.artifacts?.final;
      if(!finalRel) throw new Error('최종 산출물 경로가 없습니다.');
      const finalUrl = buildFileUrl(finalRel);
      $status.text('완료!');
      if (typeof playResult === 'function') {
        playResult(finalUrl);
      } else {
        console.log('FINAL:', finalUrl);
      }

    } catch(err){
      console.error(err);
      $status.text('실패: ' + (err && err.message ? err.message : '네트워크/서버 오류'));
    } finally {
      setBusy(false);
    }
  });

  // ---------- 전송 시 로딩 GIF ----------
  (function(){
    const LOADING_GIF = 'assets/img/loading_page.gif'; // 경로 확인
    const service = document.querySelector('#service');
    if (!service) return;
    const form    = service.querySelector('#service-upload');
    const drop    = service.querySelector('#drop-zone');
    const prompt  = service.querySelector('.upload-prompt');
    const preview = service.querySelector('#upload-preview');
    if (!form || !drop || !preview) return;
    function showLoading() {
      if (prompt) prompt.style.display = 'none';
      preview.src = LOADING_GIF;
      preview.style.display = 'block';
      preview.style.objectFit = 'cover';
      drop.classList.add('is-loading');
    }
    form.addEventListener('submit', function(){ showLoading(); }, { capture: true });
  })();

  // 초기화
  resetImgPreview();
});

function playResult(url){
  var $vid    = $('#result-video');
  var $img    = $('#upload-preview');
  var $prompt = $('#drop-zone .upload-prompt');

  if ($vid.length) {
    if ($prompt.length) $prompt.hide();
    if ($img.length) {
      $img.hide().attr('src',''); // 이미지 미리보기 감춤
    }
    // 비디오 보여주고 로드
    $vid.attr('src', url).show()[0].load();

    // 사용자가 바로 보도록 서비스 섹션으로 스크롤(선택)
    var service = document.getElementById('service');
    if (service) service.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } else {
    // 비상: 결과 자리 없으면 새 탭으로라도
    window.open(url, '_blank', 'noopener');
  }
}
