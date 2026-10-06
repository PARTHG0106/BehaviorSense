/* The local translator has one hosted client. Serialize report widgets and language
 * switches so an in-flight request cannot make the newest language fail as "busy". */
export function createTranslationQueue(request, shouldRun = () => true) {
  let tail = Promise.resolve();
  return (payload) => {
    const next = tail.then(() => {
      if (!shouldRun(payload)) throw new Error("Translation superseded by another language.");
      return request(payload);
    });
    tail = next.catch(() => {});
    return next;
  };
}
