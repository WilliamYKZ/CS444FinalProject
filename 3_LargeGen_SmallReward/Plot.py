import torch
import time
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

##############################################################################
# 1) LOAD MODELS
##############################################################################
gen_model_path = "/home/exouser/Desktop/Llama-2-7b-hf"
reward_model_path = "/home/exouser/Desktop/Qwen2.5-3B-Instruct"

# A small config to reduce memory usage (4-bit quant + auto device placement)
load_kwargs = {
    "load_in_4bit": True,
    "device_map": "auto",
    # Optionally: "torch_dtype": torch.float16,
}

# Load generation (SFT) model
gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_path)
gen_model = AutoModelForCausalLM.from_pretrained(gen_model_path, **load_kwargs)
gen_model.eval()

# Load reward model
reward_tokenizer = AutoTokenizer.from_pretrained("/home/exouser/Desktop/Llama-2-7b-hf")
reward_model = AutoModelForSequenceClassification.from_pretrained(
    reward_model_path,
    num_labels=1,
    ignore_mismatched_sizes=True,
    **load_kwargs
)
reward_model.eval()

##############################################################################
# 2) EXAMPLES (REAL-WORLD DATA) as a Dictionary
##############################################################################
examples = {
    "Example 1_100": (
        "How does IE register ActiveX controls for use in the browser? "
        "Does it just run regsvr32 for the DLL?"
    ),
    "Example 2_500": (
        "I'm trying to get the Download Folder to show on my file explorer. "
        "However on Android 9, when I use the getexternalstoragedirectory() method, "
        "it is showing self and emulated directories only and if I press \"emulated\" "
        "I cannot see more folders—it shows an empty folder. So this is how I'm getting the path: "
        "it's working fine in other Android versions but Android 9. Any guide would be appreciated \n\n"
        "``` \nval dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS).absolutePath \n```"
    ),
    "Example 3_2000": (
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
    "Example 4_4000": (
        "I have encountered a problem when trying to select data from a table in MySQL in Java by a text column that is in utf-8. "
        "The interesting thing is that with code in Python it works well, in Java it doesn't. The table looks as follows:\n"
        "```sql\n"
        "CREATE TABLE `x` (\n"
        "  `id` int(10) unsigned NOT NULL AUTO_INCREMENT,\n"
        "  `text` varchar(255) COLLATE utf8_bin NOT NULL,\n"
        "  PRIMARY KEY (`id`)\n"
        ") ENGINE=MyISAM AUTO_INCREMENT=3 DEFAULT CHARSET=utf8 COLLATE=utf8_bin;\n"
        "```\n\n"
        "The query looks like this:\n"
        "```sql\n"
        "SELECT * FROM x WHERE text = 'ěščřž'\n"
        "```\n\n"
        "The Java code that doesn't work as expected is the following:\n"
        "```java\n"
        "public class test {\n"
        "    public static void main(String [] args) {\n"
        "        java.sql.Connection conn = null;\n"
        "        System.out.println(\"SQL Test\");\n"
        "        try {\n"
        "            Class.forName(\"com.mysql.jdbc.Driver\").newInstance();\n"
        "            conn = java.sql.DriverManager.getConnection(\n"
        "                \"jdbc:mysql://127.0.0.1/x?user=root&password=root&characterSet=utf8&useUnicode=true&characterEncoding=utf-8&characterSetResults=utf8\"\n"
        "            );\n"
        "        } catch (Exception e) {\n"
        "            System.out.println(e);\n"
        "            System.exit(0);\n"
        "        }\n"
        "        System.out.println(\"Connection established\");\n"
        "        try {\n"
        "            java.sql.Statement s = conn.createStatement();\n"
        "            java.sql.ResultSet r = s.executeQuery(\"SELECT * FROM x WHERE text = 'ěščřž'\");\n"
        "            while(r.next()) {\n"
        "                System.out.println(r.getString(\"id\") + \" \" + r.getString(\"text\"));\n"
        "            }\n"
        "        } catch (Exception e) {\n"
        "            System.out.println(e);\n"
        "            System.exit(0);\n"
        "        }\n"
        "    }\n"
        "}\n"
        "```\n\n"
        "The Python code is:\n"
        "```python\n"
        "# encoding: utf8\n"
        "import MySQLdb\n"
        "conn = MySQLdb.connect(host=\"127.0.0.1\", port=3307, user=\"root\", passwd=\"root\", db=\"x\")\n"
        "cursor = conn.cursor()\n"
        "cursor.execute(\"SELECT * FROM x where text = 'ěščřž'\")\n"
        "row = cursor.fetchone()\n"
        "print(row)\n"
        "cursor.close()\n"
        "conn.close()\n"
        "```\n\n"
        "Does anybody have any suggestions what might be the cause why the Python code works and why the Java code does not? "
        "(by not working I mean not finding the desired data — the connection works fine). Many thanks."
    ),
    "Example 5_6000": (
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
    "Example 6_8000": (
        "I am an all around newbie here (new to Blender, new to Python, and new to coding in general) so please bear with me. "
        "I have a Blender script that generates a specific geometry and then renders an image. In the same script, I would then like to create a PDF file "
        "containing that image. I have two different pdf generation scripts that work perfectly fine outside of Blender (I am using Spyder) but if I run the same code in Blender, "
        "I run into problems. Here is the first one:\n\n"
        "```python\n"
        "import datetime\n"
        "from reportlab.lib.enums import TA_JUSTIFY\n"
        "from reportlab.lib.pagesizes import letter\n"
        "from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image\n"
        "from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle\n"
        "from reportlab.lib.units import mm\n"
        "import os.path\n\n"
        "formatted_date = datetime.date.today()\n"
        "date_str = str(formatted_date)\n"
        "full_name = \"Nachname, Vorname\"\n"
        "fpath = \"I:/MedTech_Projekte/NAM/Studenten/WenokorRebecca_SA/Spyder Scripts/\"\n"
        "fname = full_name + \"_\" + date_str\n"
        "fcount = 0\n"
        "fcounts = fname + \"_\" + str(fcount) + \".pdf\"\n"
        "while os.path.isfile(fcounts) == True:\n"
        "    fcount += 1\n"
        "    fcounts = fname + \"_\" + str(fcount) + \".pdf\"\n"
        "print(fcounts)\n"
        "fname = fcounts\n"
        "doc = SimpleDocTemplate(fpath + fname, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)\n"
        "Story = []\n"
        "KRIlogo = fpath + \"Klinikum_rechts_der_Isar_logo.png\"\n"
        "lg_res_x = 1920\n"
        "lg_res_y = 1080\n"
        "lg_w = 50\n"
        "lg_h = lg_w * lg_res_y / lg_res_x\n"
        "lg = Image(KRIlogo, lg_w * mm, lg_h * mm)\n"
        "lg.hAlign = 'RIGHT'\n"
        "Story.append(lg)\n"
        "wireIm = fpath + \"20170102_red_20170207-092526.png\"\n"
        "bl_res_x = 1920\n"
        "bl_res_y = 1080\n"
        "im_w = 60\n"
        "im_h = im_w * bl_res_y / bl_res_x\n"
        "im = Image(wireIm, im_w * mm, im_h * mm)\n"
        "im.hAlign = 'LEFT'\n"
        "Story.append(im)\n"
        "styles = getSampleStyleSheet()\n"
        "styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))\n"
        "ntext = '<font size=12>%s</font>' % full_name\n"
        "dtext = '<font size=12>%s</font>' % date_str\n"
        "Story.append(Paragraph(ntext, styles[\"Normal\"]))\n"
        "Story.append(Spacer(1, 12))\n"
        "Story.append(Paragraph(dtext, styles[\"Normal\"]))\n"
        "Story.append(Spacer(1, 12))\n"
        "doc.build(Story)\n"
        "```\n\n"
        "Here is the second one:\n\n"
        "```python\n"
        "import datetime\n"
        "from reportlab.pdfgen import canvas\n"
        "from reportlab.lib.units import mm\n"
        "from reportlab.lib.utils import ImageReader\n"
        "import os.path\n\n"
        "formatted_date = datetime.date.today()\n"
        "date_str = str(formatted_date)\n"
        "full_name = \"Nachname, Vorname\"\n"
        "fpath = \"I:/MedTech_Projekte/NAM/Studenten/WenokorRebecca_SA/Spyder Scripts/\"\n"
        "fname = full_name + \"_\" + date_str\n"
        "fcount = 0\n"
        "fcounts = fname + \"_\" + str(fcount) + \".pdf\"\n"
        "while os.path.isfile(fcounts) == True:\n"
        "    fcount += 1\n"
        "    fcounts = fname + \"_\" + str(fcount) + \".pdf\"\n"
        "print(fcounts)\n"
        "fname = fcounts\n"
        "wireIm = fpath + \"20170102_red_20170207-092526.png\"\n"
        "bl_res_x = 1920\n"
        "bl_res_y = 1080\n"
        "im_w = 60\n"
        "im_h = im_w * bl_res_y / bl_res_x\n"
        "WireImage = ImageReader(wireIm)\n"
        "c = canvas.Canvas(fname)\n"
        "c.drawImage(WireImage, 10, 10, width=60 * mm)\n"
        "c.showPage()\n"
        "c.save()\n"
        "```\n\n"
        "Both scripts give me pretty much the same error:\n\n"
        "```text\n"
        "Traceback (most recent call last):\n"
        "  File \"I:/MedTech_Projekte/NAM/Studenten/WenokorRebecca_SA/BLENDER CODE/2016121 9 - Present/20170109 Face Align.blend/Text.002\", line 58, in <module>\n"
        "    ...\n"
        "struct.error: unpack requires a bytes object of length 1\n"
        "Imaging Library not available, unable to import bitmaps only jpegs\n"
        "fileName='I:/MedTech_Projekte/NAM/Studenten/WenokorRebecca_SA/Spyder Scripts/Klinikum_rechts_der_Isar_logo.png'\n"
        "...\n"
        "Error: Python script fail, look in the console for now...\n"
        "```\n\n"
        "When I use jpeg instead of png, I get the following:\n\n"
        "```text\n"
        "Bibliotheken/Dokumente/Spyder Scripts/20170102_red_20170207-092526.jpeg\n"
        "Traceback (most recent call last):\n"
        "  File \"I:/MedTech_Projekte/NAM/Studenten/WenokorRebecca_SA/BLENDER CODE/2016121 9 - Present/20170109 Face Align.blend/Text.001\", line 37, in <module>\n"
        "    ...\n"
        "PermissionError: [Errno 13] Permission denied: 'Nachname, Vorname_2017-02-10_0.pdf'\n"
        "Error: Python script fail, look in the console for now...\n"
        "```\n\n"
        "A lot of online forums mention the need for PIL and/or Pillow when working with images. I don't fully understand how I would use those libraries in my code, "
        "but if the code works without them in Spyder, I don't see why it would all of a sudden need them in Blender. Any help is very much appreciated!!!\n"
        "Feel free to ask for more information if my question is not clear :)\n"
        "Thanks!"
    ),
    
    "Example 7_10000": (
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
        "        <li><span style=\"visibility:hidden;\">0</span><a href=\"/wiki/32_v._Chr.\" title=\"32 v. Chr.\">32 v. Chr.</a>: <a href=\"/wiki/Augustus\" title=\"Augustus\">Oktavian</a> lässt sich vom <a href=\"/wiki/R%C3%B6mischer_Senat\" title=\"Römischer Senat\">Senat</a> zum „Führer Italiens“ (<i><a href=\"/wiki/Dux_(Titel)\" title=\"Dux (Titel)\">dux Italiae</a></i>) ausrufen. Er erklärt <a href=\"/wiki/Kleopatra_VII.\" title=\"Kleopatra VII.\">Kleopatra</a> und damit <i><a href=\"/wiki/De_jure/de_facto\" title=\"De jure/de facto\">de facto</a></i> auch <a href=\"/wiki/Marcus_Antonius\" title=\"Marcus Antonius\">Marcus Antonius</a> den Krieg.</li>\n"
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


    "Example 8_12000": (
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
    
    "Example 9_14000": (
    "I have a problem with grpc npm-package. When I run a `npm i` it starts with: ``` > grpc@1.24.2 install C:\\RELOG\\relog\\node_modules\\grpc > node-pre-gyp install --fallback-to-build --library=static_library ``` then there are a lot errors with node-gyp, **that starts with**: ``` PS C:\\RELOG\\relog> npm i fcm-node > grpc@1.24.2 install C:\\RELOG\\relog\\node_modules\\grpc > node-pre-gyp install --fallback-to-build --library=static_library node-pre-gyp WARN Using request for node-pre-gyp https download node-pre-gyp WARN Pre-built binaries not installable for grpc@1.24.2 and node@12.16.2 (node-v72 ABI, unknown) (falling back to source compile with node-gyp) node-pre-gyp WARN Hit error connect ETIMEDOUT 104.28.23.74:443 Building the projects in this solution one at a time. To enable parallel build, please add the \"-m\" switch. win_delay_load_hook.cc WINDOWS_BUILD_WARNING.vcxproj -> C:\\RELOG\\relog\\node_modules\\grpc\\build\\Release\\\\WINDOWS_BUILD_WARNING.node address_sorting.c address_sorting_posix.c address_sorting_windows.c win_delay_load_hook.cc address_sorting.vcxproj -> C:\\RELOG\\relog\\node_modules\\grpc\\build\\Release\\\\libaddress_sorting.lib ares__close_sockets.c ares__get_hostent.c ares__read_line.c ares__timeval.c ares_cancel.c ares_create_query.c ares_data.c ares_destroy.c ares_expand_name.c ares_expand_string.c ares_fds.c ares_free_hostent.c ares_free_string.c C:\\RELOG\\relog\\node_modules\\grpc\\deps\\grpc\\third_party\\cares\\cares\\ares__get_hostent.c(141,32): warning C4996: 'inet_addr': Use inet_pton() or InetPton() instead or define _WINSOCK_DEPRECATED_NO_WARNINGS to disable deprecated API warnings [C:\\RELOG\\relog\\node_modules\\grpc\\build\\ares.vcxproj] ares_getenv.c ares_gethostbyaddr.c ares_gethostbyname.c ares_getnameinfo.c ares_getopt.c ares_getsock.c ares_init.c ares_library_init.c ares_llist.c ares_mkquery.c ares_nowarn.c ares_options.c ares_parse_a_reply.c C:\\RELOG\\relog\\node_modules\\grpc\\deps\\grpc\\third_party\\cares\\cares\\ares_gethostbyname.c(275,32): warning C4996: 'inet_addr': Use inet_pton() or InetPton() instead or define _WINSOCK_DEPRECATED_NO_WARNINGS to disable deprecated API warnings [C:\\RELOG\\relog\\node_modules\\grpc\\build\\ares.vcxproj] ares_parse_aaaa_reply.c ares_parse_mx_reply.c ares_parse_naptr_reply.c ares_parse_ns_reply.c ares_parse_ptr_reply.c C:\\RELOG\\relog\\node_modules\\grpc\\deps\\grpc\\third_party\\cares\\cares\\ares_init.c(2421,18): warning C4996: 'inet_addr': Use inet_pton() or InetPton() instead or define _WINSOCK_DEPRECATED_NO_WARNINGS to disable deprecated API warnings [C:\\RELOG\\relog\\node_modules\\grpc\\build\\ares.vcxproj] ares_parse_soa_reply.c ares_parse_srv_reply.c ares_parse_txt_reply.c ares_platform.c ares_process.c ares_query.c ares_search.c ares_send.c ares_strcasecmp.c ares_strdup.c ares_strerror.c ares_strsplit.c ares_timeout.c ares_version.c ares_writev.c bitncmp.c inet_net_pton.c inet_ntop.c windows_port.c win_delay_load_hook.cc ares.vcxproj -> C:\\RELOG\\relog\\node_modules\\grpc\\build\\Release\\\\libares.lib err_data.c a_bitstr.c a_bool.c a_d2i_fp.c a_dup.c a_enum.c a_gentm.c a_i2d_fp.c a_int.c a_mbstr.c a_object.c a_octet.c C:\\RELOG\\relog\\node_modules\\grpc\\deps\\grpc\\third_party\\boringssl\\include\\openssl\\base.h(147,1): warning C4005: 'OPENSSL_VERSION_NUMBER': macro redefinition (compiling source file ..\\deps\\grpc\\src\\boringssl\\err_data.c) [C:\\RELOG\\relog\\node_modules\\grpc\\build\\boringssl.vcxproj] C:\\Users\\crimpyhead\\AppData\\Local\\node-gyp\\Cache\\12.16.2\\include\\node\\openssl\\opensslv.h(42): message : see previous definition of 'OPENSSL_VERSION_NUMBER' (compiling source file ..\\deps\\grpc\\src\\boringssl\\err_data.c) C:\\Users\\crimpyhead\\AppData\\Local\\node-gyp\\Cache\\12.16.2\\include\\node\\openssl\\e_os2.h(171,1): warning C4005: 'OPENSSL_EXPORT': macro redefinition (compiling source file ..\\deps\\grpc\\src\\boringssl\\err_data.c) [C:\\RELOG\\relog\\node_modules\\grpc\\build\\boringssl.vcxproj] C:\\RELOG\\relog\\node_modules\\grpc\\deps\\grpc\\third_party\\boringssl\\include\\openssl\\base.h(182): message : see previous definition of 'OPENSSL_EXPORT' (compiling source file ..\\deps\\grpc\\src\\boringssl\\err_data.c) [C:\\RELOG\\relog\\node_modules\\grpc\\build\\boringssl.vcxproj] C:\\Users\\crimpyhead\\AppData\\Local\\node-gyp\\Cache\\12.16.2\\include\\node\\openssl\\ossl_typ.h(91,26): error C2371: 'EVP_MD': redefinition; different basic types (compiling source file ..\\deps\\grpc\\src\\boringssl\\err_data.c) [C:\\RELOG\\relog\\node_modules\\grpc\\build\\boringssl.vcxproj] C:\\Users\\crimpyhead\\AppData\\Local\\node-gyp\\Cache\\12.16.2\\include\\node\\openssl\\ossl_typ.h(92,30): error C2371: 'EVP_MD_CTX': redefinition; different basic types (compiling source file ..\\deps\\grpc\\src\\boringssl\\err_data.c) [C:\\RELOG\\relog\\node_modules\\grpc\\build\\boringssl.vcxproj] C:\\Users\\crimpyhead\\AppData\\Local\\node-gyp\\Cache\\12.16.2\\include\\node\\openssl\\ossl_typ.h(100,34): error C2371: 'EVP_ENCODE_CTX': redefinition; different basic types (compiling source file ..\\deps\\grpc\\src\\boringssl\\err_data.c) [C:\\RELOG\\relog\\node_modules\\grpc\\build\\boringssl.vcxproj] C:\\RELOG\\relog\\node_modules\\grpc\\deps\\grpc\\third_party\\boringssl\\include\\openssl\\base.h(308): message : see declaration of 'EVP_ENCODE_CTX' (compiling source file ..\\deps\\grpc\\src\\boringssl\\err_data.c) [C:\\RELOG\\relog\\node_modules\\grpc\\build\\boringssl.vcxproj] C:\\Users\\crimpyhead\\AppData\\Local\\node-gyp\\Cache\\12.16.2\\include\\node\\openssl\\crypto.h(231,3): error C2371: 'CRYPTO_THREADID': redefinition; different basic types (compiling source file ..\\deps\\grpc\\src\\boringssl\\err_data.c) [C:\\RELOG\\relog\\node_modules\\grpc\\build\\boringssl.vcxproj] C:\\RELOG\\relog\\node_modules\\grpc\\deps\\grpc\\src\\boringssl\\err_data.c(33,1): error C2118: negative subscript [C:\\RELOG\\relog\\node_modules\\grpc\\build\\boringssl.vcxproj] C:\\RELOG\\relog\\node_modules\\grpc\\deps\\grpc\\src\\boringssl\\err_data.c(34,1): error C2118: negative subscript [C:\\RELOG\\relog\\node_modules\\grpc\\build\\boringssl.vcxproj] C:\\RELOG\\relog\\node_modules\\grpc\\deps\\grpc\\src\\boringssl\\err_data.c(35,1): error C2118: negative subscript [C:\\RELOG\\relog\\node_modules\\grpc\\build\\boringssl.vcxproj] C:\\RELOG\\relog\\node_modules\\grpc\\deps\\grpc\\src\\boringssl\\err_data.c(36,1): error C2118: negative subscript [C:\\RELOG\\relog\\node_modules\\grpc\\build\\boringssl.vcxproj] C:\\RELOG\\relog\\node_modules\\grpc\\deps\\grpc\\src\\boringssl\\err_data.c(37,1): error C2118: negative subscript [C:\\RELOG\\relog\\node_modules\\grpc\\build\\boringssl.vcxproj] C:\\RELOG\\relog\\node_modules\\grpc\\deps\\grpc\\src\\boringssl\\err_data.c(38,1): error C2118: negative subscript [C:\\RELOG\\relog\\node_modules\\grpc\\build\\boringssl.vcxproj] C:\\RELOG\\relog\\node_modules\\grpc\\deps\\grpc\\src\\boringssl\\err_data.c(39,1): error C2118: negative subscript [C:\\RELOG\\relog\\node_modules\\grpc\\build\\boringssl.vcxproj] C:\\RELOG\\relog\\node_modules\\grpc\\deps\\grpc\\src\\boringssl\\err_data.c(40,1): error C2065: 'ERR_LIB_PKCS8': undeclared identifier [C:\\RELOG\\relog\\node_modules\\grpc\\build\\boringssl.vcxproj] C:\\RELOG\\relog\\node_modules\\grpc\\deps\\grpc\\src\\boringssl\\err_data.c(40,1): error C2057: expected constant expression [C:\\RELOG\\relog\\node_modules\\grpc\\build\\boringssl.vcxproj] C:\\RELOG\\relog\\node_modules\\grpc\\deps\\grpc\\src\\boringssl\\err_data.c(40,1): error C2466: cannot allocate an array of constant size 0 [C:\\RELOG\\relog\\node_modules\\grpc\\build\\boringssl.vcxproj] C:\\RELOG\\relog\\node_modules\\grpc\\deps\\grpc\\src\\boringssl\\err_data.c(41,1): error C2118: negative subscript [C:\\RELOG\\relog\\node_modules\\grpc\\build\\boringssl.vcxproj] C:\\RELOG\\relog\\node_modules\\grpc\\deps\\grpc\\src\\boringssl\\err_data.c(42,1): error C2118: negative subscript [C:\\RELOG\\relog\\node_modules\\grpc\\build\\boringssl.vcxproj] C:\\RELOG\\relog\\node_modules\\grpc\\deps\\grpc\\src\\boringssl\\err_data.c(43,1): error C2118: negative subscript [C:\\RELOG\\relog\\node_modules\\grpc\\build\\boringssl.vcxproj] C:\\RELOG\\relog\\node_modules\\grpc\\deps\\grpc\\src\\boringssl\\err_data.c(44,1): error C2118: negative subscript [C:\\RELOG\\relog\\node_modules\\grpc\\build\\boringssl.vcxproj] C:\\RELOG\\relog\\node_modules\\grpc\\deps\\grpc\\src\\boringssl\\err_data.c(45,1): error C2118: negative subscript [C:\\RELOG\\relog\\node_modules\\grpc\\build\\boringssl.vcxproj] C:\\RELOG\\relog\\node_modules\\grpc\\deps\\grpc\\src\\boringssl\\err_data.c(46,1): error C2118: negative subscript [C:\\RELOG\\relog\\node_modules\\grpc\\build\\boringssl.vcxproj] ``` and in the end it shows this: ``` gyp ERR! build error gyp ERR! stack Error: `C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\MSBuild\\15.0\\Bin\\MSBuild.exe` failed with exit code: 1 gyp ERR! stack at ChildProcess.onExit (C:\\Program Files\\nodejs\\node_modules\\npm\\node_modules\\node-gyp\\lib\\build.js:194:23) gyp ERR! stack at ChildProcess.emit (events.js:310:20) gyp ERR! stack at Process.ChildProcess._handle.onexit (internal/child_process.js:275:12) gyp ERR! System Windows_NT 10.0.18363 gyp ERR! command \"C:\\Program Files\\nodejs\\node.exe\" \"C:\\Program Files\\nodejs\\node_modules\\npm\\node_modules\\node-gyp\\bin\\node-gyp.js\" \"build\" \"--fallback-to-build\" \"--library=static_library\" \"--module=C:\\RELOG\\relog\\node_modules\\grpc\\src\\node\\extension_binary\\node-v72-win32-x64-unknown\\grpc_node.node\" \"--module_name=grpc_node\" \"--module_path=C:\\RELOG\\relog\\node_modules\\grpc\\src\\node\\extension_binary\\node-v72-win32-x64-unknown\" \"--napi_version=5\" \"--node_abi_napi=napi\" \"--napi_build_version=0\" \"--node_napi_label=node-v72\" gyp ERR! cwd C:\\RELOG\\relog\\node_modules\\grpc gyp ERR! node -v v12.16.2 gyp ERR! node-gyp -v v5.1.0 gyp ERR! not ok node-pre-gyp ERR! build error node-pre-gyp ERR! stack Error: Failed to execute 'C:\\Program Files\\nodejs\\node.exe C:\\Program Files\\nodejs\\node_modules\\npm\\node_modules\\node-gyp\\bin\\node-gyp.js build --fallback-to-build --library=static_library --module=C:\\RELOG\\relog\\node_modules\\grpc\\src\\node\\extension_binary\\node-v72-win32-x64-unknown\\grpc_node.node --module_name=grpc_node --module_path=C:\\RELOG\\relog\\node_modules\\grpc\\src\\node\\extension_binary\\node-v72-win32-x64-unknown --napi_version=5 --node_abi_napi=napi --napi_build_version=0 --node_napi_label=node-v72' (1) node-pre-gyp ERR! stack at ChildProcess.<anonymous> (C:\\Program Files\\nodejs\\node_modules\\grpc\\node_modules\\node-pre-gyp\\lib\\util\\compile.js:83:29) node-pre-gyp ERR! stack at ChildProcess.emit (events.js:310:20) node-pre-gyp ERR! stack at maybeClose (internal/child_process.js:1021:16) node-pre-gyp ERR! stack at Process.ChildProcess._handle.onexit (internal/child_process.js:286:5) node-pre-gyp ERR! System Windows_NT 10.0.18363 node-pre-gyp ERR! command \"C:\\Program Files\\nodejs\\node.exe\" \"C:\\RELOG\\relog\\node_modules\\grpc\\node_modules\\node-pre-gyp\\bin\\node-pre-gyp\" \"install\" \"--fallback-to-build\" \"--library=static_library\" node-pre-gyp ERR! cwd C:\\RELOG\\relog\\node_modules\\grpc node-pre-gyp ERR! node -v v12.16.2 node-pre-gyp ERR! node-pre-gyp -v v0.14.0 node-pre-gyp ERR! not ok Failed to execute 'C:\\Program Files\\nodejs\\node.exe C:\\Program Files\\nodejs\\node_modules\\npm\\node_modules\\node-gyp\\bin\\node-gyp.js build --fallback-to-build --library=static_library --module=C:\\RELOG\\relog\\node_modules\\grpc\\src\\node\\extension_binary\\node-v72-win32-x64-unknown\\grpc_node.node --module_name=grpc_node --module_path=C:\\RELOG\\relog\\node_modules\\grpc\\src\\node\\extension_binary\\node-v72-win32-x64-unknown --napi_version=5 --node_abi_napi=napi --napi_build_version=0 --node_napi_label=node-v72' (1) npm WARN babel-loader@6.4.1 requires a peer of webpack@1 || 2 || ^2.1.0-beta || ^2.2.0-rc but none is installed. You must install peer dependencies yourself. npm WARN leaflet-textpath@1.2.3 requires a peer of leaflet@^1.3.1 but none is installed. You must install peer dependencies yourself. npm WARN leaflet.markercluster@1.4.1 requires a peer of leaflet@~1.3.1 but none is installed. You must install peer dependencies yourself. npm WARN react-leaflet@1.9.1 requires a peer of leaflet@^1.3.0 but none is installed. You must install peer dependencies yourself. npm WARN react-tooltip@3.11.6 requires a peer of react@>=^16.0.0 but none is installed. You must install peer dependencies yourself. npm WARN react-tooltip@3.11.6 requires a peer of react-dom@>=^16.0.0 but none is installed. You must install peer dependencies yourself. npm WARN react-visjs-timeline@1.6.0 requires a peer of vis-timeline@^5.x but none is installed. You must install peer dependencies yourself. npm WARN optional SKIPPING OPTIONAL DEPENDENCY: grpc@1.24.2 (node_modules\\grpc): npm WARN optional SKIPPING OPTIONAL DEPENDENCY: grpc@1.24.2 install: `node-pre-gyp install --fallback-to-build --library=static_library` npm WARN optional SKIPPING OPTIONAL DEPENDENCY: Exit status 1"
    
    ),

}

##############################################################################
# 3) HELPER FUNCTIONS
##############################################################################
def generate_text_and_time(model, tokenizer, prompt, max_new_tokens=128):
    """Generate text from the model and measure how long it takes."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,  # or False if you want deterministic
        )
    torch.cuda.synchronize()
    end = time.time()

    gen_time = end - start
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return gen_time, generated_text

def reward_score_and_time(model, tokenizer, text):
    """Compute the reward model's forward pass time and extract the 1D score/logit."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(model.device)
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        out = model(**inputs)  # shape [batch, 1] typically
    torch.cuda.synchronize()
    end = time.time()

    rew_time = end - start
    score_tensor = out.logits if hasattr(out, "logits") else out[0]
    reward_value = score_tensor.squeeze().item()
    return rew_time, reward_value

##############################################################################
# 4) SINGLE RUN (ORIGINAL) - KEEP EXACTLY THE SAME
##############################################################################
single_input_lengths = []
single_gen_times = []
single_rew_times = []

output_filename = "/home/exouser/Desktop/LargeGen_SmallReward/real_data_gen_and_reward.txt"
with open(output_filename, "w", encoding="utf-8") as f:
    f.write("Real-World Examples (Single Run Data):\n\n")

    for i, (ex_name, ex_text) in enumerate(examples.items(), start=1):
        # 1) measure input length
        tokens = gen_tokenizer(ex_text)["input_ids"]
        in_len = len(tokens)
        single_input_lengths.append(in_len)

        # 2) generate text (SFT model), single run
        gen_time, gen_output = generate_text_and_time(gen_model, gen_tokenizer, ex_text)
        single_gen_times.append(gen_time)

        # 3) feed generation into reward model
        rew_time, rew_score = reward_score_and_time(reward_model, reward_tokenizer, gen_output)
        single_rew_times.append(rew_time)

        # Print to console
        print(f"[Single Run] {ex_name}: input_len={in_len}, gen_time={gen_time:.3f}s, "
              f"rew_time={rew_time:.3f}s, reward={rew_score:.4f}")

        # Save single-run details to file
        f.write(f"{ex_name}:\n")
        f.write(f"Prompt (length {in_len} tokens):\n{ex_text}\n\n")
        f.write(f"Generated response:\n{gen_output}\n\n")
        f.write(f"Reward model score: {rew_score:.4f}\n")
        f.write(f"Times => Generation: {gen_time:.3f}s, Reward: {rew_time:.3f}s\n")
        f.write("="*60 + "\n\n")


##############################################################################
# 5) MULTI RUN (10x, DROP FIRST 5, AVERAGE LAST 5) -> Use for Plot + Multrun file
##############################################################################
multi_input_lengths = []
avg_gen_times = []  # store the final average gen time per example
avg_rew_times = []  # store the final average rew time per example

# File to store the average results
multirun_filename = "/home/exouser/Desktop/LargeGen_SmallReward/Multrun.txt"
with open(multirun_filename, "w", encoding="utf-8") as f2:
    f2.write("Multi-Run Timing (10 runs, drop first 5, average last 5):\n\n")

    for i, (ex_name, ex_text) in enumerate(examples.items(), start=1):
        tokens = gen_tokenizer(ex_text)["input_ids"]
        in_len = len(tokens)
        multi_input_lengths.append(in_len)

        # Collect times over 10 runs
        gen_times_10 = []
        rew_times_10 = []

        for run_idx in range(10):
            gen_time, gen_output = generate_text_and_time(gen_model, gen_tokenizer, ex_text)
            rew_time, rew_score = reward_score_and_time(reward_model, reward_tokenizer, gen_output)
            gen_times_10.append(gen_time)
            rew_times_10.append(rew_time)

        # Drop first 5
        last_five_gen = gen_times_10[5:]
        last_five_rew = rew_times_10[5:]

        # Take average
        avg_gen = sum(last_five_gen) / len(last_five_gen)
        avg_rew = sum(last_five_rew) / len(last_five_rew)

        avg_gen_times.append(avg_gen)
        avg_rew_times.append(avg_rew)

        # Print to console
        print(f"[Multi Run Avg] {ex_name}: input_len={in_len}, "
              f"avg_gen_time={avg_gen:.3f}s, avg_rew_time={avg_rew:.3f}s")

        # Write to the Multrun file
        f2.write(f"{ex_name}:\n")
        f2.write(f"Prompt length = {in_len} tokens\n")
        f2.write(f"Averaged Generation Time = {avg_gen:.3f}s\n")
        f2.write(f"Averaged Reward Time = {avg_rew:.3f}s\n")
        f2.write("-"*60 + "\n\n")


##############################################################################
# 6) PLOT RESULTS (Using the AVERAGE times from the multi-run procedure)
##############################################################################
plt.figure(figsize=(12,6))
plt.plot(multi_input_lengths, avg_gen_times, marker='o', label="Gen Time (Avg of Last 5 / 10 Runs)")
plt.plot(multi_input_lengths, avg_rew_times, marker='o', label="Reward Time (Avg of Last 5 / 10 Runs)")
plt.xlabel("Input Prompt Length (tokens)")
plt.ylabel("Time (seconds)")
plt.title("Generation & Reward Times (10-run average, first 5 dropped)")
plt.grid(True)
plt.legend()

plot_filename = "/home/exouser/Desktop/LargeGen_SmallReward/real_data_inference_time.png"
plt.savefig(plot_filename, dpi=200)
plt.show()

print("\nDone!")
print("Single-run results file:", output_filename)
print("Multi-run average results file:", multirun_filename)
print("Plot saved to:", plot_filename)
