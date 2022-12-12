from torch import nn
import torch
from modules.encoders import LanguageEmbeddingLayer, CPC, MMILB, RNNEncoder, SubNet,GRUEncoder
from transformer import Block_fusion,Block_fusion0,make_mask,MHAtt,Block_gf,Block_split0,Block_split,filter,reconenc,Block_trans,select

class MMIM(nn.Module):
    def __init__(self, hp):
        """Construct MultiMoldal InfoMax model.
        Args: 
            hp (dict): a dict stores training and model configurations
        """
        # Base Encoders
        super(MMIM,self).__init__()
        self.hp = hp
        self.add_va = hp.add_va
        #hp.d_tout = hp.d_tin

        self.text_enc = LanguageEmbeddingLayer(hp)
        self.visual_enc = RNNEncoder(
            in_size = hp.d_vin,
            hidden_size = hp.d_vh,
            out_size = hp.d_vout,
            num_layers = hp.n_layer,
            dropout = hp.dropout_v if hp.n_layer > 1 else 0.0,
            bidirectional = hp.bidirectional
        )
        self.acoustic_enc = RNNEncoder(
            in_size = hp.d_ain,
            hidden_size = hp.d_ah,
            out_size = hp.d_aout,
            num_layers = hp.n_layer,
            dropout = hp.dropout_a if hp.n_layer > 1 else 0.0,
            bidirectional = hp.bidirectional
        )


        # Trimodal Settings
        self.fusion_prj = SubNet(
            in_size = hp.d_prjh,
            hidden_size = hp.d_prjh,
            n_class = hp.n_class,
            dropout = hp.dropout_prj
        )
        self.class1=nn.Linear(hp.d_prjh,hp.n_class)
        self.class2 = nn.Linear(hp.d_prjh, hp.n_class)
        self.class3 =nn.Linear(hp.d_prjh, hp.n_class)
        # Trimodal Settings
        self.fusion_prj2 = SubNet(
            in_size=2*hp.d_prjh,
            hidden_size=hp.d_prjh,
            n_class=hp.n_class,
            dropout=hp.dropout_prj
        )
        self.fc1=nn.Linear(768,hp.d_prjh)
        self.encoder_s1 = nn.Linear(hp.d_prjh, hp.d_prjh)
        self.encoder_s2 = nn.Linear(hp.d_prjh, hp.d_prjh)

        self.encoder_s3 = nn.Linear(hp.d_prjh, hp.d_prjh)
        self.encoder_s = nn.Linear(hp.d_prjh, hp.d_prjh)


        self.att = MHAtt(hp)

        self.block_0=Block_split0(hp)
        self.block_1 = nn.ModuleList([Block_split(hp, i) for i in range(hp.layer_fusion)])


        self.block_2 = nn.ModuleList([Block_gf(hp, i) for i in range(hp.normlayer)])
        self.block_3 = nn.ModuleList([Block_gf(hp, i) for i in range(hp.normlayer)])
        self.filter1=filter(hp)
        self.filter2 = filter(hp)
        self.filter3 = filter(hp)
        self.recenc1=reconenc(hp)
        self.recenc2=reconenc(hp)
        self.recenc3=reconenc(hp)
        #self.fumatt=nn.ModuleList([Block_trans(hp, i) for i in range(hp.layerforsemantic)])

    def forward(self, sentences, visual, acoustic, v_len, a_len, bert_sent, bert_sent_type, bert_sent_mask, y=None, mem=None):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        For Bert input, the length of text is "seq_len + 2"
        """
        enc_word = self.text_enc(sentences, bert_sent, bert_sent_type, bert_sent_mask) # (batch_size, seq_len, emb_size)
        text = enc_word[:,0,:] # (batch_size, emb_size)
        text=self.fc1(text)
        acoustic = self.acoustic_enc(acoustic, a_len)
        visual = self.visual_enc(visual, v_len)
        #print(visual.size(),text.size(),acoustic.size())
        x_mask=make_mask(visual)
        y_mask=make_mask(text)
        z_mask=make_mask(acoustic)
        x,y,z=select(visual,text,acoustic)

        x_shared = self.encoder_s1(x)
        #x_shared=self.encoder_s(x_shared)
        y_shared = self.encoder_s2(y)
        #y_shared = self.encoder_s(y_shared)
        z_shared = self.encoder_s3(z)
        #z_shared = self.encoder_s(z_shared)
        #s_shared=(x_shared+y_shared+z_shared)/3
        x=self.att(x, x, y_shared,None)
        y = self.att(y, y, y_shared,None)
        z = self.att(z, z, y_shared,None)
        x0=x+y_shared
        y0=2*y_shared
        z0=z+y_shared
        x,y1,y2,z,pre1,pre2,pre3,pre4=self.block_0(x,x_mask,y,y_mask,z,z_mask)
        for i, dec in enumerate(self.block_1):
            x_m, y_m, z_m = None, None, None


            x, y1,y2, z,pre1,pre2,pre3,pre4 = dec(x, x_m, y1,y2, y_m, z, z_m,pre1,pre2,pre3,pre4)
        y=y2
        for i, dec in enumerate(self.block_2):
            x_m, y_m, z_m = None, None, None


            fusion = dec( y, y_m, x, x_m,z, z_m)


        #fusion, preds = self.fusion_prj(torch.cat([x,y,z], dim=1))
        fusion, preds1 = self.fusion_prj(fusion)
        x_sin,x_dif=self.filter1(x,x_shared,y_shared,z_shared,fusion)
        y_sin,y_dif=self.filter2(y,y_shared,x_shared,z_shared,fusion)
        z_sin,z_dif=self.filter3(z,z_shared,x_shared,y_shared,fusion)
        x_pre=self.class1(x_sin)
        y_pre=self.class2(y_sin)
        z_pre=self.class3(z_sin)
        for i, dec in enumerate(self.block_3):
            x_m, y_m, z_m = None, None, None


            fusion2 = dec( y_sin, y_m, x_sin, x_m,z_sin, z_m)

        '''x2 = self.att(x_sin, x_sin, fusion, None)
        y2 = self.att(y_sin, y_sin, fusion, None)
        z2 = self.att(z_sin, z_sin, fusion, None)'''
        #semantic=torch.cat((x2,y2,z2),dim=1)
        #semantic=x2+y2+z2
        #xrec=self.recenc1(x_sin,x_shared,fusion)
        #yrec=self.recenc2(y_sin,y_shared,fusion)
        #zrec=self.recenc3(z_sin,z_shared,fusion)
        semantic = torch.cat((fusion2,fusion), dim=1)
        '''for i, dec in enumerate(self.fumatt):


            semantic = dec(semantic)'''
        f1, preds = self.fusion_prj2(semantic)
        return preds,pre1,pre2,pre3,pre4,x_shared,y_shared,z_shared,fusion,preds1,x_pre,y_pre,z_pre,x_dif,y_dif,z_dif,x,y,z,x_sin,y_sin,z_sin,text,visual,acoustic,x0,y0,z0
