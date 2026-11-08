import { createByteBpe } from './byte-bpe.js';
import { createCitySlots } from './city-slots.js';
import { InputError, PicoboiEngine } from './inference.js';

export class Picoboi10M extends PicoboiEngine {
  constructor(manifest, tokenizer, buffer, aliases) {
    super(manifest, createByteBpe(tokenizer), buffer);
    this.aliases = aliases;
    this.slotIds = Array.from({ length: 8 }, (_, i) => this.ids[`<city_${i + 1}>`]);
  }

  prepare(messages) {
    if (!messages.length || messages.length % 2 !== 1) throw new InputError('Start a new chat and try again.');
    let turns = messages;
    while (turns.length) {
      this.parser = createCitySlots(this.aliases);
      const tokens = [this.ids['<bos>']];
      for (const [i, message] of turns.entries()) {
        if (message.role !== (i % 2 ? 'assistant' : 'user')) throw new InputError('Start a new chat and try again.');
        tokens.push(this.ids[`<${message.role}>`]);
        for (const segment of this.parser.encode(message.content, message.role)) {
          const ids = segment.slot ? [this.slotIds[segment.slot - 1]] : this.tokenizer.encode(segment.text);
          for (const id of ids) tokens.push(id);
        }
        tokens.push(this.ids['<end_turn>']);
      }
      tokens.push(this.ids['<assistant>']);
      if (tokens.length + 48 <= this.config.train_context) return { tokens, dropped: messages.length - turns.length };
      turns = turns.slice(2);
    }
    throw new InputError('That message is too long for the 512-token context.');
  }

  forbiddenTokens() {
    return new Set(this.tokenizer.tokens.flatMap((token, id) => {
      if (token === '<eos>' || token === '<end_turn>') return [];
      const slot = this.slotIds.indexOf(id) + 1;
      if (slot) return this.parser.resolve(slot) ? [] : [id];
      return id < 17 || token.startsWith('<unused_') ? [id] : [];
    }));
  }

  decode(ids, streaming) {
    let text = '';
    let ordinary = [];
    for (const id of ids) {
      const binding = this.parser.resolve(this.slotIds.indexOf(id) + 1);
      if (binding) {
        text += this.tokenizer.decode(ordinary, false) + binding.name;
        ordinary = [];
      } else ordinary.push(id);
    }
    return (text + this.tokenizer.decode(ordinary, streaming)).trimStart();
  }
}
