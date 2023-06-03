package wspr

import (
	"fmt"
	"io"
	"time"

	whisper "github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
)

type Model struct {
	model whisper.Model
}

type Segment struct {
	Text       string
	Start, End time.Duration
}

type opt struct {
	lang string
	maxToken uint
}

type Opt func(*opt)

func New(path string) (*Model, error) {
	model, err := whisper.New(path)
	if err != nil {
		return nil, fmt.Errorf("failed to load model in %s: %w", path, err)
	}

	return &Model{
		model: model,
	}, nil
}


func WithLang(lang string) Opt {
	return func(o *opt) {
		o.lang = lang
	}
}

func WithMaxTokenPerSegment(n uint) Opt {
	return func(o *opt) {
		o.maxToken = n
	}
}

func (m Model) Transcribe(monoWav []float32, opts ...Opt) (chan Segment, error) {
	var options opt
	for _, ofn := range opts {
		ofn(&options)
	}

	var ch = make(chan Segment)
	ctx, err := m.model.NewContext()

	if err != nil {
		return nil, fmt.Errorf("failed to init new transcription process: %w", err)
	}

	if options.lang != "" {
		ctx.SetLanguage(options.lang)
	}

	if options.maxToken != 0 {
		ctx.SetMaxTokensPerSegment(options.maxToken)
	}

	ctx.ResetTimings()

	err = ctx.Process(monoWav, nil)

	if err != nil {
		return nil, fmt.Errorf("failed to start new transcription process: %w", err)
	}

	go func() {
		defer close(ch)
		for {
			s, err := ctx.NextSegment()
			if err == io.EOF {
				return
			}

			var seg Segment
			seg.Text = s.Text
			seg.Start = s.Start
			seg.End = s.End

			ch <- seg
		}
	}()

	return ch, nil
}
