// Project facts are UI content, separate from model generation and model history.
export function getProjectReply(input) {
  const question = input.toLowerCase().trim()
    .replace(/[?!.,]+$/u, '').replace(/\s+/gu, ' ');
  const subject = '(?:you|picoboi|pico|this (?:model|llm|chatbot|webpage|website|project))';
  const madeQuestion = new RegExp(`^(?:please )?(?:tell me )?who (?:made|built|created|developed) ${subject}$`, 'u');
  const creatorQuestion = /^(?:who is|who's|who’s) (?:your|the) (?:creator|maker|developer)(?: of (?:picoboi|this project|this model))?$/u;
  const attributionQuestion = /^(?:who is (?:behind|the person behind) (?:picoboi|this project)|(?:who are|who're|who’re) you made by)$/u;
  if (!madeQuestion.test(question) && !creatorQuestion.test(question) && !attributionQuestion.test(question)) return null;
  return {
    prefix: 'I was made by ',
    label: '@farilyaz',
    url: 'https://twitter.com/farilyaz',
    suffix: '.',
  };
}
