import os
import torch
import numpy as np
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
from openai import OpenAI


NUM_RUNS = 5
use_random_seeds = True 

# A small config to reduce memory usage (4-bit quant + auto device placement)
load_kwargs = {
    "load_in_4bit": True,
    "device_map": "auto",
    # Optionally: "torch_dtype": torch.float16,
}

##################################
# 1. Load the SFT model & tokenizer
##################################
# sft_model_name_or_path = "/home/exouser/Desktop/DeepSeek-R1-Distill-Qwen-7B"
# sft_tokenizer = AutoTokenizer.from_pretrained(sft_model_name_or_path)
# sft_model = AutoModelForCausalLM.from_pretrained(
#     sft_model_name_or_path,
#     **load_kwargs
# )
# sft_model.eval()

##################################
# 2. Load the Reward Model & tokenizer
##################################
reward_model_name_or_path = "/home/exouser/Desktop/1_PPO/checkpoint-5000_peft_stack-exchange-paired_rmts__100000_2e-05/checkpoint-12660"
reward_tokenizer = AutoTokenizer.from_pretrained("/home/exouser/Desktop/Llama-2-7b-hf")

reward_model = AutoModelForCausalLM.from_pretrained(
    reward_model_name_or_path,
    num_labels=1,
    ignore_mismatched_sizes=True,
    **load_kwargs
)
reward_model.eval()

# If you have GPU(s) available, move models to GPU for faster inference.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# sft_model.to(device)
reward_model.to(device)

##################################
# 3. Define example prompts
##################################
example_prompts = {
    "Example 1_500": (
        "I'm trying to get the Download Folder to show on my file explorer. "
        "However on Android 9, when I use the getexternalstoragedirectory() method, "
        "it is showing self and emulated directories only and if I press \"emulated\" "
        "I cannot see more folders—it shows an empty folder. So this is how I'm getting the path: "
        "it's working fine in other Android versions but Android 9. Any guide would be appreciated \n\n"
        "``` \nval dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS).absolutePath \n```"
    ),
    "Example 2_2000": (
        "I'm currently trying to extend [a model](https://github.com/microsoft/MASS) that is based on FairSeq/PyTorch. "
        "During training I need to train two encoders: one with the target sample, and the original one with the source sample. "
        "So the current forward function looks like this:\n"
        "```python\n"
        "def forward(self, src_tokens=None, src_lengths=None, prev_output_tokens=None, **kwargs):\n"
        "    encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)\n"
        "    decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)\n"
        "    return decoder_out\n"
        "```\n"
        "And based on [this idea](https://github.com/golsun/SpaceFusion) I want something like this:\n"
        "```python\n"
        "def forward_test(self, src_tokens=None, src_lengths=None, prev_output_tokens=None, **kwargs):\n"
        "    encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)\n"
        "    decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)\n"
        "    return decoder_out\n\n"
        "def forward_train(self, src_tokens=None, src_lengths=None, prev_output_tokens=None, **kwargs):\n"
        "    encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)\n"
        "    autoencoder_out = self.encoder(tgt_tokens, src_lengths=src_lengths, **kwargs)\n"
        "    concat = some_concatination_func(encoder_out, autoencoder_out)\n"
        "    decoder_out = self.decoder(prev_output_tokens, encoder_out=concat, **kwargs)\n"
        "    return decoder_out\n"
        "```\n"
        "Is there any way to do this?\n\n"
        "Edit: These are the constraints that I have, since I need to extend *FairseqEncoderDecoderModel*:\n"
        "```python\n"
        "@register_model('transformer_mass')\n"
        "class TransformerMASSModel(FairseqEncoderDecoderModel):\n"
        "    def __init__(self, encoder, decoder):\n"
        "        super().__init__(encoder, decoder)\n"
        "```\n"
        "Edit 2: The parameters passed to the forward function in Fairseq can be altered by implementing your own Criterion, "
        "see for example [CrossEntropyCriterion](https://github.com/pytorch/fairseq/blob/master/fairseq/criterions/cross_entropy.py#L28), "
        "where `sample['net_input']` is passed to the `__call__` function of the model, which invokes the `forward` method."
    ),

    "Example 3_6000": (
        "```java\n"
        "public AddressBookApp(){\n"
        "    frame = new JFrame(\"Address Book\");\n"
        "    frame.setSize(500, 400);\n"
        "    frame.setDefaultCloseOperation(frame.EXIT_ON_CLOSE);\n"
        "    panel = new JPanel();\n"
        "    panel.setBackground(Color.gray);\n"
        "    panel.setLayout(null);\n"
        "    frame.add(panel);\n"
        "    frame.setVisible(true);\n"
        "    JMenuBar menubar = new JMenuBar();\n"
        "    frame.setJMenuBar(menubar);\n"
        "    JMenu file = new JMenu(\"File\");\n"
        "    menubar.add(file);\n"
        "    JMenuItem insert = new JMenuItem(\"Import\");\n"
        "    file.add(insert);\n"
        "    insert.addActionListener(this);\n"
        "    JMenuItem export = new JMenuItem(\"Export\");\n"
        "    file.add(export);\n"
        "    export.addActionListener(this);\n"
        "    JMenuItem exit = new JMenuItem(\"Exit\");\n"
        "    file.add(exit);\n"
        "    exit.addActionListener(this);\n"
        "    Font f = new Font(\"Helvetica\", Font.BOLD, 10);\n"
        "    btnadd = new JButton(\"Add\");\n"
        "    btnadd.setFont(f);\n"
        "    btnadd.setBounds(200, 250, 80, 20);\n"
        "    panel.add(btnadd);\n"
        "    btnprev = new JButton(\"Previous\");\n"
        "    btnprev.setBounds(40, 250, 80, 20);\n"
        "    btnprev.setFont(f);\n"
        "    btnprev.addActionListener(this);\n"
        "    panel.add(btnprev);\n"
        "    btnnxt = new JButton(\"Next\");\n"
        "    btnnxt.setBounds(120, 250, 80, 20);\n"
        "    btnnxt.setFont(f);\n"
        "    btnnxt.addActionListener(this);\n"
        "    panel.add(btnnxt);\n"
        "    btndel = new JButton(\"Delete\");\n"
        "    btndel.setBounds(280, 250, 80, 20);\n"
        "    btndel.setFont(f);\n"
        "    panel.add(btndel);\n"
        "    btnclear = new JButton(\"Clear\");\n"
        "    btnclear.setBounds(360, 250, 80, 20);\n"
        "    btnclear.setFont(f);\n"
        "    btnclear.addActionListener(this);\n"
        "    panel.add(btnclear);\n"
        "    txtname = new JTextField(\"\");\n"
        "    txtname.setBounds(210, 40, 160, 20);\n"
        "    txtname.setFont(f);\n"
        "    panel.add(txtname);\n"
        "    txtnum = new JTextField(\"\");\n"
        "    txtnum.setBounds(210, 70, 160, 20);\n"
        "    txtnum.setFont(f);\n"
        "    panel.add(txtnum);\n"
        "    txtmob = new JTextField(\"\");\n"
        "    txtmob.setBounds(210, 100, 160, 20);\n"
        "    txtmob.setFont(f);\n"
        "    panel.add(txtmob);\n"
        "    txtadd1 = new JTextField(\"\");\n"
        "    txtadd1.setBounds(210, 130, 160, 20);\n"
        "    txtadd1.setFont(f);\n"
        "    panel.add(txtadd1);\n"
        "    lblname = new JLabel(\"Name\");\n"
        "    lblname.setBounds(160, 40, 160, 20);\n"
        "    lblname.setFont(f);\n"
        "    panel.add(lblname);\n"
        "    lblnum = new JLabel(\"Number\");\n"
        "    lblnum.setBounds(160, 70, 160, 20);\n"
        "    lblnum.setFont(f);\n"
        "    panel.add(lblnum);\n"
        "    lblmob = new JLabel(\"Mobile\");\n"
        "    lblmob.setBounds(160, 100, 160, 20);\n"
        "    lblmob.setFont(f);\n"
        "    panel.add(lblmob);\n"
        "    lbladd1 = new JLabel(\"Address \");\n"
        "    lbladd1.setBounds(160, 130, 160, 20);\n"
        "    lbladd1.setFont(f);\n"
        "    panel.add(lbladd1);\n"
        "}\n\n"
        "public static void main(String[] args) {\n"
        "    AddressBookApp ab = new AddressBookApp();\n"
        "}\n\n"
        "public void actionPerformed(ActionEvent e) {\n"
        "    if (e.getActionCommand().equals(\"Exit\"))\n"
        "        System.exit(0);\n"
        "    else if (e.getActionCommand().equals(\"Import\")) {\n"
        "        importContacts();\n"
        "    } else if (e.getActionCommand().equals(\"Export\")); {\n"
        "        exportContacts();\n"
        "    }\n"
        "    if (e.getSource() == btnnxt) {\n"
        "        nextContact();\n"
        "    } else if (e.getSource() == btnprev) {\n"
        "        prevContact();\n"
        "    }\n"
        "}\n\n"
        "public void importContacts() {\n"
        "    try{\n"
        "        BufferedReader fileSize = new BufferedReader(new FileReader(\"../files/example.buab\"));\n"
        "        BufferedReader importContacts = new BufferedReader(new FileReader(\"../files/example.buab\"));\n"
        "        int i = 0;\n"
        "        String contacts;\n"
        "        while (( fileSize.readLine()) != null) {\n"
        "            details.add(importContacts.readLine());\n"
        "            i++;\n"
        "        }\n"
        "        fileSize.close();\n"
        "        int x = 0;\n"
        "        int y = 0;\n"
        "        for (x = 0, y = 0; x < details.size(); x++, y++) {\n"
        "            if (y == 4) { y = 0; }\n"
        "            if (y == 0) { name.add(details.get(x)); }\n"
        "            if (y == 1) { phone.add(details.get(x)); }\n"
        "            if (y == 2) { mobile.add(details.get(x)); }\n"
        "            if (y == 3) { address.add(details.get(x)); }\n"
        "        }\n"
        "    } catch (IOException ioe) {\n"
        "        ioe.printStackTrace();\n"
        "    }\n"
        "    txtname.setText(name.get(0));\n"
        "    txtnum.setText(phone.get(0));\n"
        "    txtmob.setText(mobile.get(0));\n"
        "    txtadd1.setText(address.get(0));\n"
        "}\n\n"
        "public void exportContacts() {\n"
        "    FileOutputStream file;\n"
        "    PrintStream out;\n"
        "    try {\n"
        "        file = new FileOutputStream(\"../files/example.buab\", true);\n"
        "        out = new PrintStream(file);\n"
        "        out.println(txtname.getText());\n"
        "        out.println(txtnum.getText());\n"
        "        out.println(txtmob.getText());\n"
        "        out.println(txtadd1.getText());\n"
        "        System.err.println(\"\");\n"
        "        out.close();\n"
        "    } catch (Exception e) {\n"
        "        System.err.println(\"Error in writing to file\");\n"
        "    }\n"
        "}\n\n"
        "public void nextContact() {\n"
        "    if(index < details.size() - 1) {\n"
        "        index++;\n"
        "        txtname.setText(name.get(index));\n"
        "        txtnum.setText(phone.get(index));\n"
        "        txtmob.setText(mobile.get(index));\n"
        "        txtadd1.setText(address.get(index));\n"
        "    }\n"
        "    importContacts();\n"
        "}\n\n"
        "public void prevContact() {\n"
        "    if (index > 0) {\n"
        "        index--;\n"
        "        txtname.setText(name.get(index));\n"
        "        txtnum.setText(phone.get(index));\n"
        "        txtmob.setText(mobile.get(index));\n"
        "        txtadd1.setText(address.get(index));\n"
        "    }\n"
        "    importContacts();\n"
        "}\n"
        "```"
    ),

    
    "Example 4_10000": (
        "Problem Description:\n"
        "--------------------\n"
        "I am looking for a way to access the `li`-elements between two specific heading-tags only (e.g., from 2nd `h3` to 3rd `h3` or from 3rd `h3` to next `h4`) in order to create a table of historical events listed on <https://de.wikipedia.org/wiki/1._Januar> structured along the criteria mentioned in the headings. A major problem (for me ...) is that - other than the `h1`-heading - the subtitles of the lower levels have no `className` or `id`.\n\n"
        
        "Sample of HTML:\n"
        "---------------\n"
        "```html\n"
        "<div class=\"mw-parser-output\">\n"
        "    [...] \n"
        "    </h3>\n"
        "    <ul>\n"
        "        <li><a href=\"/wiki/153_v._Chr.\" title=\"153 v. Chr.\">153 v. Chr.</a>: Die <a href=\"/wiki/Consulat\" title=\"Consulat\">Konsuln</a> der <a href=\"/wiki/R%C3%B6mische_Republik\" title=\"Römische Republik\">römischen Republik</a> beginnen ihre Amtszeit erstmals am 1. Januar statt am 1. März; daher ist der 1. Januar heute der Jahresanfang.</li>\n"
        "        <li><span style=\"visibility:hidden;\">0</span><a href=\"/wiki/45_v._Chr.\" title=\"45 v. Chr.\">45 v. Chr.</a>: <a href=\"/wiki/Kalenderreform_des_Gaius_Iulius_Caesar\" title=\"Kalenderreform des Gaius Iulius Caesar\">Caesars Reform</a> des <a href=\"/wiki/R%C3%B6mischer_Kalender\" title=\"Römischer Kalender\">römischen Kalenders</a> endet. Dieser wird ab 2. Januar 709 <a href=\"/wiki/Ab_urbe_condita_(Chronologie)\" title=\"Ab urbe condita (Chronologie)\">a. u. c.</a> durch den <a href=\"/wiki/Julianischer_Kalender\" title=\"Julianischer Kalender\">julianischen Kalender</a> ersetzt.</li>\n"
        "        <li><span style=\"visibility:hidden;\">0</span><a href=\"/wiki/32_v._Chr.\" title=\"32 v. Chr.\">32 v. Chr.</a>: <a href=\"/wiki/Augustus\" title=\"Augustus\">Oktavian</a> l\u00e4sst sich vom <a href=\"/wiki/R%C3%B6mischer_Senat\" title=\"Römischer Senat\">Senat</a> zum \"Führer Italiens\" (<i><a href=\"/wiki/Dux_(Titel)\" title=\"Dux (Titel)\">dux Italiae</a></i>) ausrufen. Er erkl\u00e4rt <a href=\"/wiki/Kleopatra_VII.\" title=\"Kleopatra VII.\">Kleopatra</a> und damit <i><a href=\"/wiki/De_jure/de_facto\" title=\"De jure/de facto\">de facto</a></i> auch <a href=\"/wiki/Marcus_Antonius\" title=\"Marcus Antonius\">Marcus Antonius</a> den Krieg.</li>\n"
        "    </ul>\n"
        "    [...] \n"
        "    </ul>\n"
        "    <h4><span id=\"Inkrafttreten_von_Gesetzen_und_Staatsvertr.C3.A4gen\"></span><span class=\"mw-headline\" id=\"Inkrafttreten_von_Gesetzen_und_Staatsverträgen\">Inkrafttreten von Gesetzen und Staatsverträgen</span><span class=\"mw-editsection\"><span class=\"mw-editsection-bracket\">[</span> <a href=\"/w/index.php?title=1._Januar&amp;veaction=edit&amp;section=3\" class=\"mw-editsection-visualeditor\" title=\"Abschnitt bearbeiten: Inkrafttreten von Gesetzen und Staatsverträgen\">Bearbeiten</a><span class=\"mw-editsection-divider\"> | </span> <a href=\"/w/index.php?title=1._Januar&amp;action=edit&amp;section=3\" title=\"Abschnitt bearbeiten: Inkrafttreten von Gesetzen und Staatsverträgen\">Quelltext bearbeiten</a><span class=\"mw-editsection-bracket\">]</span></span> </h4>\n"
        "    <p><i>Der 1. Januar wird oft für das Inkrafttreten von Gesetzen und Staatsverträgen verwendet. Das gilt unter anderem für:</i> </p>\n"
        "    <ul>\n"
        "        <li><a href=\"/wiki/1812\" title=\"1812\">1812</a>: das <i><a href=\"/wiki/Allgemeines_b%C3%BCrgerliches_Gesetzbuch\" title=\"Allgemeines bürgerliches Gesetzbuch\">Allgemeine bürgerliche Gesetzbuch</a></i> <i>(ABGB)</i> in den <a href=\"/wiki/Habsburgermonarchie#Erblande\" title=\"Habsburgermonarchie\">habsburgischen Erblanden</a>.</li>\n"
        "    </ul>\n"
        "    [...] \n"
        "    </h4>\n"
        "    <p><i>Folgende Staaten erhalten am 1. Januar ihre Unabhängigkeit:</i> </p>\n"
        "    <ul>\n"
        "        [...] \n"
        "    </ul>\n"
        "    <h3><span class=\"mw-headline\" id=\"Wirtschaft\">Wirtschaft</span><span class=\"mw-editsection\"><span class=\"mw-editsection-bracket\">[</span><a href=\"/w/index.php?title=1._Januar&amp;veaction=edit&amp;section=6\" class=\"mw-editsection-visualeditor\" title=\"Abschnitt bearbeiten: Wirtschaft\">Bearbeiten</a> <span class=\"mw-editsection-divider\"> | </span><a href=\"/w/index.php?title=1._Januar&amp;action=edit&amp;section=6\" title=\"Abschnitt bearbeiten: Wirtschaft\">Quelltext bearbeiten</a><span class=\"mw-editsection-bracket\">]</span></span> </h3>\n"
        "    <h4><span class=\"mw-headline\" id=\"Wichtige_Ereignisse_in_der_Weltwirtschaft\">Wichtige Ereignisse in der Weltwirtschaft</span><span class=\"mw-editsection\"><span class=\"mw-editsection-bracket\">[</span><a href=\"/w/index.php?title=1._Januar&amp;veaction=edit&amp;section=7\" class=\"mw-editsection-visualeditor\" title=\"Abschnitt bearbeiten: Wichtige Ereignisse in der Weltwirtschaft\">Bearbeiten</a><span class=\"mw-editsection-divider\"> | </span><a href=\"/w/index.php?title=1._Januar&amp;action=edit&amp;section=7\" title=\"Abschnitt bearbeiten: Wichtige Ereignisse in der Weltwirtschaft\">Quelltext bearbeiten</a> <span class=\"mw-editsection-bracket\">]</span> </span> </h4>\n"
        "    <ul>\n"
        "        <li><a href=\"/wiki/1780\" title=\"1780\">1780</a>: In <a href=\"/wiki/Geschichte_Bratislavas\" title=\"Geschichte Bratislavas\">Preßburg</a> erscheint die erste ungarische Zeitung <i>Magyar hírmondó</i> („Ungarischer Kurier“).</li>\n"
        "    ```\n\n"
        
        "So far, I only managed to access **all** the `li`-elements (more than 1000!) that are not part of the table of contents with the following code:\n\n"
        
        "Experimental Code Example:\n"
        "--------------------------\n"
        "```vba\n"
        "Sub HistoricalEvents_Test()\n"
        "    Dim http As Object, html As New MSHTML.HTMLDocument\n"
        "    Dim oLiList As MSHTML.IHTMLDOMChildrenCollection\n"
        "    Dim data As String\n"
        "    Dim r As Integer\n"
        "    Dim oWord As Object, oWordDoc As Object\n"
        "    Dim wordApp As New Word.Application\n"
        "    Dim iFirstRow As Integer, iLastRow As Integer\n\n"
        "    Set http = CreateObject(\"MSXML2.XMLHTTP\")\n"
        "    http.Open \"GET\", \"https://de.wikipedia.org/wiki/1._Januar\", False\n"
        "    http.send\n"
        "    html.body.innerHTML = http.responseText\n\n"
        "    Dim lLiResultList As Long\n"
        "    Dim lLiResultLoop As Long\n"
        "    Set oLiList = html.querySelectorAll(\"#toc ~ ul li\")\n\n"
        "    For lLiResultLoop = 0 To oLiList.Length - 1\n"
        "        Dim oLiChild As Object\n"
        "        Set oLiChild = oIlList.Item(lLilResultLoop)\n"
        "        data = oLiChild.innerText\n"
        "        'data = data & vbCrLf & oLiChild.innerText\n"
        "        Range(\"B\" & lLiResultLoop +1).Value = data\n"
        "        data = vbNullString\n"
        "    Next lLiResultLoop\n\n"
        "    Dim j As Long\n"
        "    Dim Ws As Worksheet\n"
        "    Dim rngDB As Range\n\n"
        "    Set Ws = ActiveSheet\n"
        "    Set oWord = CreateObject(\"Word.Application\")\n"
        "    Set oWordDoc = oWord.Documents.Open(\"D:\\Jahrestage Geschichte.docx\")\n"
        "    iFirstRow = 1 ' \"Ws.Cells(1, 2).End(xlDown).Row\" used to work fine but suddenly gives same as iLastRow!\n"
        "    'Debug.Print iFirstRow\n"
        "    iLastRow = Ws.Cells(ActiveSheet.Rows.Count, \"B\").End(xlUp).Row\n"
        "    'Debug.Print iLastRow\n"
        "    oWord.Visible = True\n\n"
        "    With wordApp\n"
        "        With Ws\n"
        "            Set rngDB = Ws.Range(.Cells(iFirstRow, 2), .Cells(iLastRow, 2))\n"
        "        End With\n"
        "        rngDB.Cut\n"
        "        oWord.Selection.PasteSpecial DataType:=wdPasteText\n"
        "        oWord.Selection.TypeParagraph\n"
        "        oWord.Selection = \"\"\n"
        "    End With\n\n"
        "    oWordDoc.Close savechanges:=True\n"
        "    wordApp.Quit 'it doesn't :(\n"
        "End Sub\n"
        "```\n\n"
        
        "Description of General Idea/Final Project:\n"
        "-----------------------------------------\n"
        "The final project is supposed to have a worksheet for every month, each containing a table with a row for every day of the respective month and columns for the different categories according to the (sub-)titles. The Word-output in the code is just an early-stage by-product and something I will round off only if/when the main problem can be solved.\n\n"
        
        "Further Remarks:\n"
        "---------------\n"
        "This is my first contribution on SO. I'm an absolute beginner when it comes to VBA and web-scraping (or any kind of coding, scripting or programming for that matter), but I kind of got sucked into it and spent the better part of my winter holiday just to figure out the above code. I wouldn't have been able to accomplish even that poor piece of scripting without the invaluable knowledge shared with noobs like me by the cracks of SO. I've tried out various approaches but I always got stuck at some point, VBA triggering runtime errors and often Excel crashing. In particular, I wasn't able to implement the `nextSibling`/`previousSibling` methods or the `nodeName` selector successfully which I figure might be a promising approach to the problem. So any help or hint would be greatly appreciated!\n\n"
        
        "Working Solution:\n"
        "-----------------\n"
        "Thanks to the feedback on my question I finally managed to figure out a solution that does the job, although maybe not in the most elegant way. The only remaining problem is that strangely the `li-elements` of the last column are duplicated. So if anyone has a clue how to deal with that ...\n\n"
        "```vba\n"
        "Sub SliceHtmlByHeaderTypes4()\n"
        "    Dim http As Object, html As MSHTML.HTMLDocument\n"
        "    Dim sh As Worksheet\n"
        "    Set sh = ThisWorkbook.ActiveSheet\n\n"
        "    Set http = CreateObject(\"MSXML2.XMLHTTP\")\n"
        "    Set html = New MSHTML.HTMLDocument\n\n"
        "    http.Open \"GET\", \"https://de.wikipedia.org/wiki/1._Januar\", False\n"
        "    http.send\n"
        "    html.body.innerHTML = http.responseText\n\n"
        "    Dim hNodeList As Object\n"
        "    Dim startPos As Long, endPos As Long\n"
        "    Dim s As Integer, e As Integer\n\n"
        "    Set hNodeList = html.querySelectorAll(\"#toc ~ h2, #toc ~ h3, #toc ~ h4\")\n"
        "    Debug.Print hNodeList.Length\n\n"
        "    Do While s < hNodeList.Length - 1\n"
        "        http.Open \"GET\", \"https://de.wikipedia.org/wiki/1._Januar\", False\n"
        "        http.send\n"
        "        html.body.innerHTML = http.responseText\n"
        "        Set hNodeList = html.querySelectorAll(\"#toc ~ h2, #toc ~ h3, #toc ~ h4\")\n\n"
        "        startPos = InStr(html.body.outerHTML, hNodeList.Item(s).outerHTML)\n"
        "        endPos = InStr(html.body.outerHTML, hNodeList.Item(s + 1).outerHTML)\n\n"
        "        If startPos > 0 And endPos > 0 And endPos > startPos Then\n"
        "            Dim strS As String\n"
        "            strS = Mid$(html.body.outerHTML, startPos, endPos - startPos + 1)\n"
        "        Else\n"
        "            MsgBox \"Problem slicing string\"\n"
        "            Stop\n"
        "            Exit Sub\n"
        "        End If\n\n"
        "        Dim liList As Object\n"
        "        html.body.innerHTML = strS\n"
        "        Set liList = html.getElementsByTagName(\"li\")\n\n"
        "        If liList.Length > 0 Then\n"
        "            Dim i As Integer\n"
        "            Dim liText As String\n"
        "            Dim lc As Integer\n"
        "            Dim liRange As Range\n"
        "            lc = (Cells(2, Columns.Count).End(xlToLeft).Column) + 1\n"
        "            Set liRange = sh.Range(Cells(2, lc), Cells(2, lc))\n\n"
        "            For i = 0 To liList.Length - 1\n"
        "                On Error Resume Next\n"
        "                liText = liList.Item(i).innerText\n"
        "                liRange.Value = liRange.Value & liText & vbCrLf\n"
        "                liText = vbNullString\n"
        "            Next i\n\n"
        "            strS = vbNullString\n"
        "            startPos = 0\n"
        "            endPos = 0\n"
        "            hNodeList = \"\"\n"
        "            i = 0\n"
        "        End If\n"
        "        s = s + 1\n"
        "    Loop\n"
        "End Sub\n"
        "```"
    ),


    "Example 5_12000": (
        "Problem Description:\n"
        "--------------------\n"
        "OK now I know that this question has been asked before several times on SO but none of the answers have worked for me. I am attempting to create a custom preference for my project. More specifically it is a preference with a [`HorizontalListView`](http://www.dev-smart.com/archives/34) attached directly underneath it. I basically created it by modifying [this code](http://robobunny.com/wp/2011/08/13/android-seekbar-preference/) for a `SeekBarPreference` (which I am also using and is working fine). My `ListViewPreference` is located in exactly the same folder as the `SeekBarPreference` (which, as I said is having no problem) but I am constantly getting a `ClassNotFoundException` (see logcat below). Here is my `ListViewPreference` class:\n\n"
        
        "```java\n"
        "package com.example.ebookreader;\n\n"
        "import android.content.Context;\n"
        "import android.content.res.TypedArray;\n"
        "import android.preference.Preference;\n"
        "import android.util.AttributeSet;\n"
        "import android.util.Log;\n"
        "import android.view.LayoutInflater;\n"
        "import android.view.View;\n"
        "import android.view.ViewGroup;\n"
        "import android.view.ViewGroup.LayoutParams;\n"
        "import android.view.ViewParent;\n"
        "import android.widget.RelativeLayout;\n"
        "import android.widget.TextView;\n\n"
        
        "public class ListViewPreference extends Preference {\n"
        "    private final String TAG = getClass().getName();\n"
        "    private static final String ROBOBUNNYNS = \"http://robobunny.com\";\n"
        "    private static final int DEFAULT_VALUE = 50;\n"
        "    private int mCurrentValue;\n"
        "    private String mUnitsLeft = \"\";\n"
        "    private String mUnitsRight = \"\";\n"
        "    private HorizontalListView mListView;\n"
        "    private TextView mStatusText;\n\n"
        
        "    public ListViewPreference(Context context, AttributeSet attrs) {\n"
        "        super(context, attrs);\n"
        "        initPreference(context, attrs);\n"
        "    }\n\n"
        
        "    public ListViewPreference(Context context, AttributeSet attrs, int defStyle) {\n"
        "        super(context, attrs, defStyle);\n"
        "        initPreference(context, attrs);\n"
        "    }\n\n"
        
        "    private void initPreference(Context context, AttributeSet attrs) {\n"
        "        setValuesFromXml(attrs);\n"
        "        mListView = new HorizontalListView(context, attrs);\n"
        "        LayoutParams params = mListView.getLayoutParams();\n"
        "        params.width = LayoutParams.MATCH_PARENT;\n"
        "        params.height = LayoutParams.WRAP_CONTENT;\n"
        "        mListView.setLayoutParams(params);\n"
        "    }\n\n"
        
        "    private void setValuesFromXml(AttributeSet attrs) {\n"
        "        mUnitsLeft = getAttributeStringValue(attrs, ROBOBUNNYNS, \"unitsLeft\", \"\");\n"
        "        String units = getAttributeStringValue(attrs, ROBOBUNNYNS, \"units\", \"\");\n"
        "        mUnitsRight = getAttributeStringValue(attrs, ROBOBUNNYNS, \"unitsRight\", units);\n"
        "    }\n\n"
        
        "    private String getAttributeStringValue(AttributeSet attrs, String namespace, String name, String defaultValue) {\n"
        "        String value = attrs.getAttributeValue(namespace, name);\n"
        "        if (value == null) value = defaultValue;\n"
        "        return value;\n"
        "    }\n\n"
        
        "    @Override\n"
        "    protected View onCreateView(ViewGroup parent) {\n"
        "        RelativeLayout layout = null;\n"
        "        try {\n"
        "            LayoutInflater mInflater = (LayoutInflater) getContext()\n"
        "                .getSystemService(Context.LAYOUT_INFLATER_SERVICE);\n"
        "            layout = (RelativeLayout) mInflater.inflate(\n"
        "                R.layout.horizontal_list_view_preference, parent, false);\n"
        "        } catch (Exception e) {\n"
        "            Log.e(TAG, \"Error creating seek bar preference\", e);\n"
        "        }\n"
        "        return layout;\n"
        "    }\n\n"
        
        "    @Override\n"
        "    public void onBindView(View view) {\n"
        "        super.onBindView(view);\n"
        "        try {\n"
        "            // move our seekbar to the new view we've been given\n"
        "            ViewParent oldContainer = mListView.getParent();\n"
        "            ViewGroup newContainer = (ViewGroup) view.findViewById(R.id.listViewPrefBarContainer);\n"
        "            if (oldContainer != newContainer) {\n"
        "                // remove the seekbar from the old view\n"
        "                if (oldContainer != null) {\n"
        "                    ((ViewGroup) oldContainer).removeView(mListView);\n"
        "                }\n"
        "                // remove the existing seekbar (there may not be one) and add ours\n"
        "                newContainer.removeAllViews();\n"
        "                newContainer.addView(mListView, ViewGroup.LayoutParams.FILL_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);\n"
        "            }\n"
        "        } catch (Exception ex) {\n"
        "            Log.e(TAG, \"Error binding view: \" + ex.toString());\n"
        "        }\n"
        "        updateView(view);\n"
        "    }\n\n"
        
        "    /**\n"
        "     * Update a SeekBarPreference view with our current state\n"
        "     *\n"
        "     * @param view\n"
        "     */\n"
        "    protected void updateView(View view) {\n"
        "        try {\n"
        "            RelativeLayout layout = (RelativeLayout) view;\n"
        "            mStatusText = (TextView) layout.findViewById(R.id.listViewPrefValue);\n"
        "            mStatusText.setText(String.valueOf(mCurrentValue));\n"
        "            mStatusText.setMinimumWidth(30);\n"
        "            TextView unitsRight = (TextView) layout.findViewById(R.id.listViewPrefUnitsRight);\n"
        "            unitsRight.setText(mUnitsRight);\n"
        "            TextView unitsLeft = (TextView) layout.findViewById(R.id.listViewPrefUnitsLeft);\n"
        "            unitsLeft.setText(mUnitsLeft);\n"
        "        } catch (Exception e) {\n"
        "            Log.e(TAG, \"Error updating seek bar preference\", e);\n"
        "        }\n"
        "    }\n\n"
        
        "    @Override\n"
        "    protected Object onGetDefaultValue(TypedArray ta, int index) {\n"
        "        int defaultValue = ta.getInt(index, DEFAULT_VALUE);\n"
        "        return defaultValue;\n"
        "    }\n\n"
        
        "    @Override\n"
        "    protected void onSetInitialValue(boolean restoreValue, Object defaultValue) {\n"
        "        if (restoreValue) {\n"
        "            mCurrentValue = getPersistedInt(mCurrentValue);\n"
        "        } else {\n"
        "            int temp = 0;\n"
        "            try {\n"
        "                temp = (Integer) defaultValue;\n"
        "            } catch (Exception ex) {\n"
        "                Log.e(TAG, \"Invalid default value: \" + defaultValue.toString());\n"
        "            }\n"
        "            persistInt(temp);\n"
        "            mCurrentValue = temp;\n"
        "        }\n"
        "    }\n"
        "}\n"
        "```\n\n"
        
        "For most people this problem is the result of not having a constructor with parameters `Context` and `AttributeSet` but as you can see, I clearly have it. Here is my XML file where the error keeps on occurring:\n\n"
        
        "```xml\n"
        "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
        "<PreferenceScreen xmlns:android=\"http://schemas.android.com/apk/res/android\"\n"
        "    xmlns:custom=\"http://schemas.android.com/apk/res/com.example.ebookreader\"\n"
        "    android:background=\"@drawable/wallpaper\" >\n\n"
        "    <android:PreferenceCategory\n"
        "        android:key=\"device_settings\"\n"
        "        android:title=\"Device Settings\" >\n\n"
        "        <android:CheckBoxPreference\n"
        "            android:key=\"wifi\"\n"
        "            android:summary=\"Enable or Disable Wi-Fi\"\n"
        "            android:title=\"Wi-Fi\" />\n\n"
        "        <android:CheckBoxPreference\n"
        "            android:key=\"bluetooth\"\n"
        "            android:summary=\"Enable or Disable Bluetooth\"\n"
        "            android:title=\"Bluetooth\" />\n\n"
        "        <android:CheckBoxPreference\n"
        "            android:key=\"autosync\"\n"
        "            android:summary=\"Enable or Disable AutoSync\"\n"
        "            android:title=\"AutoSync\" />\n\n"
        "        <custom:SeekBarPreference\n"
        "            android:id=\"@+id/brightness_adjust\"\n"
        "            android:defaultValue=\"100\"\n"
        "            android:key=\"brightness\"\n"
        "            android:max=\"200\"\n"
        "            android:summary=\"Adjust Brightness Levels\"\n"
        "            android:title=\"Brightness\" />\n"
        "    </android:PreferenceCategory>\n\n"
        "    <android:PreferenceCategory\n"
        "        android:key=\"account_settings\"\n"
        "        android:title=\"Account Settings\" >\n\n"
        "        <custom:ListViewPreference\n"
        "            android:id=\"@+id/font_selector\"\n"
        "            android:key=\"font\"\n"
        "            android:title=\"Font\"\n"
        "            android:summary=\"Edit Font\" />\n"
        "    </android:PreferenceCategory>\n"
        "</PreferenceScreen>\n"
        "```\n\n"
        
        "Below is my full Logcat:\n\n"
        
        "```text\n"
        "03-14 17:53:14.290: E/AndroidRuntime(449): FATAL EXCEPTION: main\n"
        "03-14 17:53:14.290: E/AndroidRuntime(449): java.lang.RuntimeException: Unable to start activity ComponentInfo{com.example.ebookreader/com.example.ebookreader.SettingsActivity}: android.view.InflateException: Binary XML file line #40: Error inflating class ListViewPreference\n"
        "03-14 17:53:14.290: E/AndroidRuntime(449):     at android.app.ActivityThread.performLaunchActivity(ActivityThread.java:1736)\n"
        "03-14 17:53:14.290: E/AndroidRuntime(449):     at android.app.ActivityThread.handleLaunchActivity(ActivityThread.java:1752)\n"
        "03-14 17:53:14.290: E/AndroidRuntime(449):     at android.app.ActivityThread.access$1500(ActivityThread.java:123)\n"
        "03-14 17:53:14.290: E/AndroidRuntime(449):     at android.app.ActivityThread$H.handleMessage(ActivityThread.java:993)\n"
        "03-14 17:53:14.290: E/AndroidRuntime(449):     at android.os.Handler.dispatchMessage(Handler.java:99)\n"
        "03-14 17:53:14.290: E/AndroidRuntime(449):     at android.os.Looper.loop(Looper.java:126)\n"
        "03-14 17:53:14.290: E/AndroidRuntime(449):     at android.app.ActivityThread.main(ActivityThread.java:3997)\n"
        "03-14 17:53:14.290: E/AndroidRuntime(449):     at java.lang.reflect.Method.invokeNative(Native Method)\n"
        "03-14 17:53:14.290: E/AndroidRuntime(449):     at java.lang.reflect.Method.invoke(Method.java:491)\n"
        "03-14 17:53:14.290: E/AndroidRuntime(449):     at com.android.internal.os.ZygoteInit$MethodAndArgsCaller.run(ZygoteInit.java:841)\n"
        "03-14 17:53:14.290: E/AndroidRuntime(449):     at com.android.internal.os.ZygoteInit.main(ZygoteInit.java:599)\n"
        "03-14 17:53:14.290: E/AndroidRuntime(449):     at dalvik.system.NativeStart.main(Native Method)\n"
        "03-14 17:53:14.290: E/AndroidRuntime(449): Caused by: android.view.InflateException: Binary XML file line #40: Error inflating class ListViewPreference\n"
        "03-14 17:53:14.290: E/AndroidRuntime(449):     at android.preference.GenericInflater.createItemFromTag(GenericInflater.java:441)\n"
        "03-14 17:53:14.290: E/AndroidRuntime(449):     at android.preference.GenericInflater.rInflate(GenericInflater.java:481)\n"
        "03-14 17:53:14.290: E/AndroidRuntime(449):     at android.preference.GenericInflater.rInflate(GenericInflater.java:493)\n"
        "03-14 17:53:14.290: E/AndroidRuntime(449):     at android.preference.GenericInflater.inflate(GenericInflater.java:326)\n"
        "03-14 17:53:14.290: E/AndroidRuntime(449):     at android.preference.GenericInflater.inflate(GenericInflater.java:263)\n"
        "03-14 17:53:14.290: E/AndroidRuntime(449):     at android.preference.PreferenceManager.inflateFromResource(PreferenceManager.java:269)\n"
        "03-14 17:53:14.290: E/AndroidRuntime(449):     at android.preference.PreferenceActivity.addPreferencesFromResource(PreferenceActivity.java:1333)\n"
        "03-14 17:53:14.290: E/AndroidRuntime(449):     at com.example.ebookreader.SettingsActivity.onCreate(SettingsActivity.java:31)\n"
        "03-14 17:53:14.290: E/AndroidRuntime(449):     at android.app.Instrumentation.callActivityOnCreate(Instrumentation.java:1048)\n"
        "03-14 17:53:14.290: E/AndroidRuntime(449):     at android.app.ActivityThread.performLaunchActivity(ActivityThread.java:1700)\n"
        "03-14 17:53:14.290: E/AndroidRuntime(449):     ... 11 more\n"
        "03-14 17:53:14.290: E/AndroidRuntime(449): Caused by: java.lang.ClassNotFoundException: android.preference.ListViewPreference in loader dalvik.system.PathClassLoader[/data/app/com.example.ebookreader-2.apk]\n"
        "03-14 17:53:14.290: E/AndroidRuntime(449):     at dalvik.system.PathClassLoader.findClass(PathClassLoader.java:251)\n"
        "03-14 17:53:14.290: E/AndroidRuntime(449):     at java.lang.ClassLoader.loadClass(ClassLoader.java:548)\n"
        "03-14 17:53:14.290: E/AndroidRuntime(449):     at java.lang.ClassLoader.loadClass(ClassLoader.java:508)\n"
        "03-14 17:53:14.290: E/AndroidRuntime(449):     at android.preference.GenericInflater.createItem(GenericInflater.java:375)\n"
        "03-14 17:53:14.290: E/AndroidRuntime(449):     at android.preference.GenericInflater.onCreateItem(GenericInflater.java:417)\n"
        "03-14 17:53:14.290: E/AndroidRuntime(449):     at android.preference.GenericInflater.createItemFromTag(GenericInflater.java:428)\n"
        "03-14 17:53:14.290: E/AndroidRuntime(449):     ... 20 more\n"
        "```\n\n"
        
        "Any help would be greatly appreciated."
    ),
    


}

##############################################################################
# HELPER FUNCTIONS
##############################################################################

# Initialize OpenAI client
client = OpenAI(
    api_key="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"
)

def generate_response(prompt, max_new_tokens=2048, run_index=0):
    """
    Generate a response using GPT-4 given a prompt.
    """
    try:
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            stop=None
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error generating response: {e}")
        return f"[Generated text for run {run_index}]\nPrompt was: {prompt}"

def get_reward_score(text_segment):
    """
    Pass the text segment into the reward model and return a scalar reward score.
    Adapt this to your actual reward model's forward pass / head.
    """
    inputs = reward_tokenizer(text_segment, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = reward_model(**inputs)
        # Placeholder: just use the mean of logits as a 'score' for illustration.
        logits = outputs.logits
        score = logits.mean().item()
    return score

##############################################################################
# MAIN SCRIPT
##############################################################################

# Create an output directory
output_dir = "/home/exouser/Desktop/8_RewardScore_DeepSeekModel"
os.makedirs(output_dir, exist_ok=True)

# This will hold final results of shape:
# all_scores[prompt_index] = [ [scores_run1], [scores_run2], ..., [scores_runN] ]
# where each [scores_runX] is length 10 (one reward per segment)
all_scores = []
example_names = list(example_prompts.keys())

# For each example prompt
for idx, (example_name, prompt_text) in enumerate(example_prompts.items(), start=1):
    runs_scores_for_prompt = []  # Will be a list of 10-element lists

    for run_idx in range(NUM_RUNS):
        # Optional: set random seeds differently for each run
        if use_random_seeds:
            seed_val = random.randint(0, 999999)
            torch.manual_seed(seed_val)
            np.random.seed(seed_val)
            random.seed(seed_val)

        # 1) Generate the full answer from your SFT model (or other generative model)
        generated_text = generate_response(prompt_text, run_index=run_idx)

        # 2) Tokenize the entire *generated* response
        tokens = reward_tokenizer.encode(generated_text)
        total_tokens = len(tokens)

        # If fewer than 10 tokens, skip (or store placeholders)
        if total_tokens < 10:
            print(f"Response too short (<10 tokens) for {example_name}, run {run_idx}. Skipping.")
            # You can store empty or repeated zero if you prefer
            runs_scores_for_prompt.append([0.0]*10)
            continue

        # 3) Split the *generated response* into 10 equal segments
        segment_size = total_tokens // 10
        segment_scores = []

        for segment_idx in range(10):
            if segment_idx < 9:
                end_idx = (segment_idx + 1) * segment_size
            else:
                # The last segment goes up to the end
                end_idx = total_tokens
            
            partial_tokens = tokens[:end_idx]
            partial_text = reward_tokenizer.decode(partial_tokens, skip_special_tokens=True)

            # 4) Compute reward for this partial text
            score = get_reward_score(partial_text)
            segment_scores.append(score)

        runs_scores_for_prompt.append(segment_scores)

    # After all runs for this prompt, store in all_scores
    all_scores.append(runs_scores_for_prompt)

# 5) Now, compute mean and std across runs for each segment
#    and plot them for each prompt.
for idx, runs_scores_for_prompt in enumerate(all_scores):
    prompt_name = example_names[idx]
    runs_scores_for_prompt = np.array(runs_scores_for_prompt)  # shape: (NUM_RUNS, 10)

    # Mean and std dev across runs => shape (10,)
    mean_scores = runs_scores_for_prompt.mean(axis=0)
    std_scores  = runs_scores_for_prompt.std(axis=0)

    # Plot
    x_axis = range(1, 11)
    plt.figure(figsize=(8, 5))
    plt.errorbar(x_axis, mean_scores, yerr=std_scores, fmt='-o', capsize=4)
    plt.title(f"Reward Scores for {prompt_name} (N={NUM_RUNS} runs)")
    plt.xlabel("Segment Number (1..10)")
    plt.ylabel("Reward Score")
    plt.xticks(range(1, 11))
    plt.grid(True)

    # Save the figure
    plot_path = os.path.join(output_dir, f"Reward_Scores_for_{prompt_name}.png")
    plt.savefig(plot_path)
    plt.close()

    # Optionally, you could print or save numeric results to a CSV
    # for debugging or record-keeping
    txt_path = os.path.join(output_dir, f"{prompt_name}_stats.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Mean reward scores per segment:\n{mean_scores}\n\n")
        f.write(f"Std dev of reward scores per segment:\n{std_scores}\n\n")
        f.write(f"All run data:\n{runs_scores_for_prompt}\n")

print("All done! Check your output directory for plots and text files.")