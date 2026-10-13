import { fTranslate } from '../i18n';
import { MimeTypeAudio } from '$lib/enums';

/**
 * AudioRecorder - Browser-based audio recording with MediaRecorder API
 *
 * This class provides a complete audio recording solution using the browser's MediaRecorder API.
 * It handles microphone access, recording state management, and audio format optimization.
 *
 * **Features:**
 * - Automatic microphone permission handling
 * - Audio enhancement (echo cancellation, noise suppression, auto gain)
 * - Multiple format support with fallback (WAV, WebM, MP4, AAC)
 * - Real-time recording state tracking
 * - Proper cleanup and resource management
 */
export class AudioRecorder {
  private vMediaRecorder: MediaRecorder | null = null;
  private aAudioChunks: Blob[] = [];
  private vStream: MediaStream | null = null;
  private vRecordingState: boolean = false;

  async mStartRecording(): Promise<void> {
    try {
      this.vStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });

      this.mInitializeRecorder(this.vStream);

      this.aAudioChunks = [];
      // Start recording with a small timeslice to ensure we get data
      this.vMediaRecorder!.start(100);
      this.vRecordingState = true;
    } catch (vError) {
      console.error('Failed to start recording:', vError);
      throw new Error(fTranslate('text_532670b3a185'));
    }
  }

  async mStopRecording(): Promise<Blob> {
    return new Promise((pResolve, pReject) => {
      const cRecorder = this.vMediaRecorder;
      const cChunks = this.aAudioChunks;
      const cStream = this.vStream;

      if (!cRecorder || cRecorder.state === 'inactive') {
        pReject(new Error(fTranslate('text_8a387b8b0fa8')));
        return;
      }

      // Detach instance state right away so a new startRecording can take over without race
      this.vMediaRecorder = null;
      this.aAudioChunks = [];
      this.vStream = null;
      this.vRecordingState = false;

      cRecorder.onstop = () => {
        const cAudioBlob = new Blob(cChunks, {
          type: cRecorder.mimeType || MimeTypeAudio.WAV
        });

        if (cStream) {
          for (const cTrack of cStream.getTracks()) {
            cTrack.stop();
          }
        }

        pResolve(cAudioBlob);
      };

      cRecorder.onerror = (pEvent) => {
        console.error('Recording error:', pEvent);

        if (cStream) {
          for (const cTrack of cStream.getTracks()) {
            cTrack.stop();
          }
        }

        pReject(new Error(fTranslate('text_83aeb329ae94')));
      };

      cRecorder.stop();
    });
  }

  mIsRecording(): boolean {
    return this.vRecordingState;
  }

  mCancelRecording(): void {
    const cRecorder = this.vMediaRecorder;
    const cStream = this.vStream;

    this.vMediaRecorder = null;
    this.aAudioChunks = [];
    this.vStream = null;
    this.vRecordingState = false;

    if (cRecorder && cRecorder.state !== 'inactive') {
      // Drop the original handlers so the pending stop event does not touch the instance
      cRecorder.onstop = null;
      cRecorder.onerror = null;
      cRecorder.stop();
    }

    if (cStream) {
      for (const cTrack of cStream.getTracks()) {
        cTrack.stop();
      }
    }
  }

  private mInitializeRecorder(pStream: MediaStream): void {
    const dOptions: MediaRecorderOptions = {};

    if (MediaRecorder.isTypeSupported(MimeTypeAudio.WAV)) {
      dOptions.mimeType = MimeTypeAudio.WAV;
    } else if (MediaRecorder.isTypeSupported(MimeTypeAudio.WEBM_OPUS)) {
      dOptions.mimeType = MimeTypeAudio.WEBM_OPUS;
    } else if (MediaRecorder.isTypeSupported(MimeTypeAudio.WEBM)) {
      dOptions.mimeType = MimeTypeAudio.WEBM;
    } else if (MediaRecorder.isTypeSupported(MimeTypeAudio.MP4)) {
      dOptions.mimeType = MimeTypeAudio.MP4;
    } else {
      console.warn('No preferred audio format supported, using default');
    }

    this.vMediaRecorder = new MediaRecorder(pStream, dOptions);

    this.vMediaRecorder.ondataavailable = (pEvent) => {
      if (pEvent.data.size > 0) {
        this.aAudioChunks.push(pEvent.data);
      }
    };

    this.vMediaRecorder.onstop = () => {
      this.vRecordingState = false;
    };

    this.vMediaRecorder.onerror = (pEvent) => {
      console.error('MediaRecorder error:', pEvent);
      this.vRecordingState = false;
    };
  }
}

export async function fConvertToWav(pAudioBlob: Blob): Promise<Blob> {
  try {
    if (pAudioBlob.type.includes('wav')) {
      return pAudioBlob;
    }

    const cArrayBuffer = await pAudioBlob.arrayBuffer();

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const cAudioContext = new (window.AudioContext || (window as any).webkitAudioContext)();

    try {
      const cAudioBuffer = await cAudioContext.decodeAudioData(cArrayBuffer);
      return fAudioBufferToWav(cAudioBuffer);
    } finally {
      cAudioContext.close();
    }
  } catch (vError) {
    console.error('Failed to convert audio to WAV:', vError);
    return pAudioBlob;
  }
}

function fAudioBufferToWav(pBuffer: AudioBuffer): Blob {
  const cLength = pBuffer.length;
  const cNumberOfChannels = pBuffer.numberOfChannels;
  const cSampleRate = pBuffer.sampleRate;
  const cBytesPerSample = 2; // 16-bit
  const cBlockAlign = cNumberOfChannels * cBytesPerSample;
  const cByteRate = cSampleRate * cBlockAlign;
  const cDataSize = cLength * cBlockAlign;
  const cBufferSize = 44 + cDataSize;

  const cArrayBuffer = new ArrayBuffer(cBufferSize);
  const cView = new DataView(cArrayBuffer);

  const fWriteString = (pOffset: number, pString: string) => {
    for (let vI = 0; vI < pString.length; vI++) {
      cView.setUint8(pOffset + vI, pString.charCodeAt(vI));
    }
  };

  fWriteString(0, 'RIFF'); // ChunkID
  cView.setUint32(4, cBufferSize - 8, true); // ChunkSize
  fWriteString(8, 'WAVE'); // Format
  fWriteString(12, 'fmt '); // Subchunk1ID
  cView.setUint32(16, 16, true); // Subchunk1Size
  cView.setUint16(20, 1, true); // AudioFormat (PCM)
  cView.setUint16(22, cNumberOfChannels, true); // NumChannels
  cView.setUint32(24, cSampleRate, true); // SampleRate
  cView.setUint32(28, cByteRate, true); // ByteRate
  cView.setUint16(32, cBlockAlign, true); // BlockAlign
  cView.setUint16(34, 16, true); // BitsPerSample
  fWriteString(36, 'data'); // Subchunk2ID
  cView.setUint32(40, cDataSize, true); // Subchunk2Size

  // Cache channel arrays, write PCM via Int16Array (native little-endian, matches WAV)
  const aChannels: Float32Array[] = new Array(cNumberOfChannels);
  for (let vC = 0; vC < cNumberOfChannels; vC++) {
    aChannels[vC] = pBuffer.getChannelData(vC);
  }

  const cPcm = new Int16Array(cArrayBuffer, 44, cLength * cNumberOfChannels);
  let vP = 0;
  for (let vI = 0; vI < cLength; vI++) {
    for (let vC = 0; vC < cNumberOfChannels; vC++) {
      let vS = aChannels[vC][vI];
      if (vS > 1) vS = 1;
      else if (vS < -1) vS = -1;
      cPcm[vP++] = vS * 0x7fff;
    }
  }

  return new Blob([cArrayBuffer], { type: MimeTypeAudio.WAV });
}

/**
 * Create a File object from audio blob with timestamp-based naming
 * @param pAudioBlob - The audio blob to wrap
 * @param pFilename - Optional custom filename
 * @returns File object with appropriate name and metadata
 */
export function fCreateAudioFile(pAudioBlob: Blob, pFilename?: string): File {
  const cTimestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const cExtension = pAudioBlob.type.includes('wav') ? 'wav' : 'mp3';
  const cDefaultFilename = `recording-${cTimestamp}.${cExtension}`;

  return new File([pAudioBlob], pFilename || cDefaultFilename, {
    type: pAudioBlob.type,
    lastModified: Date.now()
  });
}

/**
 * Check if audio recording is supported in the current browser
 * @returns True if MediaRecorder and getUserMedia are available
 */
export function fIsAudioRecordingSupported(): boolean {
  return !!(
    typeof navigator !== 'undefined' &&
    navigator.mediaDevices &&
    typeof navigator.mediaDevices.getUserMedia === 'function' &&
    typeof window !== 'undefined' &&
    window.MediaRecorder
  );
}
