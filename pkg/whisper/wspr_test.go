package wspr

import (
	"os"
	"testing"

	wav "github.com/go-audio/wav"
	. "github.com/smartystreets/goconvey/convey"
)

func TestTranscribeEng(t *testing.T) {
	Convey("it can transcribe Eng", t, func() {
		t.Log(os.Getwd())
		w, err := New("../../models/ggml-tiny.en.bin")
		So(err, ShouldBeNil)

		t.Log("open wav file")
		fh, err := os.Open("../../testdata/jfk.wav")
		So(err, ShouldBeNil)

		defer fh.Close()

		t.Log("decode wav file")
		dec := wav.NewDecoder(fh)
		buf, err := dec.FullPCMBuffer()
		So(err, ShouldBeNil)

		t.Log("start transcription")
		segs, err := w.Transcribe(buf.AsFloat32Buffer().Data, WithMaxTokenPerSegment(6))
		So(err, ShouldBeNil)

		for s := range segs {
			t.Logf("%+v", s)
		}
	})
}
