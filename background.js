browser.runtime.onMessage.addListener((message, sender) => {
    if (message.action == "DELIVER_IMAGE") {
        return browser.runtime.sendNativeMessage("inference", {
            image: message.image
        });
    }
});

browser.action.onClicked.addListener(() => {
  browser.sidebarAction.open();
});