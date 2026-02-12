#target photoshop

(function () {
    var pythonPath = prompt("Python executable path:", "python");
    if (!pythonPath) {
        alert("Cancelled.");
        return;
    }

    var scriptPath = File.openDialog("Select business_card_extractor.py");
    if (!scriptPath) {
        alert("Cancelled.");
        return;
    }

    var inputFile = File.openDialog("Select scan input (image / TIFF / PDF)");
    if (!inputFile) {
        alert("Cancelled.");
        return;
    }

    var outputFolder = Folder.selectDialog("Select output folder for card PNGs");
    if (!outputFolder) {
        alert("Cancelled.");
        return;
    }

    var cmd = '"' + pythonPath + '" "' + scriptPath.fsName + '" "' + inputFile.fsName + '" -o "' + outputFolder.fsName + '"';
    var result = app.system(cmd);

    alert("Business card extraction finished.\n\nOutput: " + outputFolder.fsName + "\n\nCommand output:\n" + result);
})();
