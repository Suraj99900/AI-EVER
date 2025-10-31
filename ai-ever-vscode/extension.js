const vscode = require("vscode");
const { fetch } = require("undici");

let lastRequestTime = 0;
const REQUEST_INTERVAL = 30000; // 30 seconds throttle

// 🔹 Helper: call backend
async function callAIBackend(payload, statusBarItem) {
    try {
        statusBarItem.text = "$(sync~spin) AI-EVER running...";
        statusBarItem.show();

        const res = await fetch("http://127.0.0.1:5000/complete?current_session=4", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        const data = await res.json();

        statusBarItem.text = "✅ AI-EVER done!";
        setTimeout(() => statusBarItem.hide(), 1500);

        if (data?.status !== "success") {
            vscode.window.showErrorMessage("AI-EVER error: " + (data?.message || "Unknown error"));
            return null;
        }

        return data;
    } catch (err) {
        statusBarItem.hide();
        vscode.window.showErrorMessage("AI-EVER backend error: " + err.message);
        return null;
    }
}

// 🔹 Inline provider with throttling
const inlineProvider = {
    async provideInlineCompletionItems(document, position) {
        const codeBefore = document.getText(new vscode.Range(new vscode.Position(0, 0), position));
        if (!codeBefore.trim()) return [];

        const now = Date.now();
        if (now - lastRequestTime < REQUEST_INTERVAL) return []; // throttle
        lastRequestTime = now;

        const statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
        const data = await callAIBackend({ context: codeBefore, task: "completion" }, statusBarItem);

        if (data?.completion?.trim()) {
            return [
                {
                    insertText: data.completion,
                    range: new vscode.Range(position, position),
                },
            ];
        }
        return [];
    }
};

// 🔹 Activate extension
function activate(context) {
    console.log("AI-EVER extension activated ✅");

    const statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
    context.subscriptions.push(statusBarItem);

    // Commands: Manual completion
    context.subscriptions.push(
        vscode.commands.registerCommand("ai-ever-vscode.completeCode", async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) return;

            const text = editor.selection.isEmpty
                ? editor.document.getText()
                : editor.document.getText(editor.selection);

            const data = await callAIBackend({ context: text, task: "completion" }, statusBarItem);

            if (data?.completion) {
                editor.edit(editBuilder => {
                    if (editor.selection.isEmpty) {
                        editBuilder.insert(editor.selection.start, data.completion);
                    } else {
                        editBuilder.replace(editor.selection, data.completion);
                    }
                });
            }
        }),

        vscode.commands.registerCommand("ai-ever-vscode.fixCode", async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) return;

            const selection = editor.selection;
            if (selection.isEmpty) {
                return vscode.window.showErrorMessage("Select some buggy code to fix.");
            }

            const buggyCode = editor.document.getText(selection);
            const data = await callAIBackend({ context: buggyCode, task: "bug_fix" }, statusBarItem);

            if (data?.completion) {
                editor.edit(editBuilder => editBuilder.replace(selection, data.completion));
            }
        })
    );

    // Inline completion provider
    context.subscriptions.push(
        vscode.languages.registerInlineCompletionItemProvider({ pattern: "**" }, inlineProvider)
    );

    // 🔹 Listen to typing and trigger inline suggestions (throttled)
    context.subscriptions.push(
        vscode.workspace.onDidChangeTextDocument(async event => {
            const editor = vscode.window.activeTextEditor;
            if (!editor || editor.document !== event.document) return;

            const now = Date.now();
            if (now - lastRequestTime < REQUEST_INTERVAL) return; // throttle
            lastRequestTime = now;

            // Trigger inline suggestion
            await vscode.commands.executeCommand("editor.action.inlineSuggest.trigger");
        })
    );
}

// 🔹 Deactivate extension
function deactivate() {}

module.exports = { activate, deactivate };
