# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
    graph_encoder
)
from fairseq.modules import AdaptiveSoftmax

DEFAULT_MAX_SOURCE_POSITIONS = 1e5
DEFAULT_MAX_TARGET_POSITIONS = 1e5


@register_model('lstm')
class LSTMModel(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder, graph_encoder=None):
        super().__init__(encoder, decoder, graph_encoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-freeze-embed', action='store_true',
                            help='freeze encoder embeddings')
        parser.add_argument('--encoder-hidden-size', type=int, metavar='N',
                            help='encoder hidden size')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='number of encoder layers')
        parser.add_argument('--encoder-bidirectional', action='store_true',
                            help='make all layers of encoder bidirectional')
        # graph settings
        parser.add_argument('--n-graph-layers', type=int, metavar='N')
        parser.add_argument('--graph-encoder-embed-dim', type=int, metavar='N')
        parser.add_argument('--graph-in-dropout', type=float, metavar='N')
        parser.add_argument('--graph-out-dropout', type=float, metavar='N')

        # decoder settings
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-freeze-embed', action='store_true',
                            help='freeze decoder embeddings')
        parser.add_argument('--decoder-hidden-size', type=int, metavar='N',
                            help='decoder hidden size')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='number of decoder layers')
        parser.add_argument('--decoder-out-embed-dim', type=int, metavar='N',
                            help='decoder output embedding dimension')
        parser.add_argument('--decoder-attention', type=str, metavar='BOOL',
                            help='decoder attention')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion')
        parser.add_argument('--share-decoder-input-output-embed', default=False,
                            action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', default=False, action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')

        # Granular dropout settings (if not specified these default to --dropout)
        parser.add_argument('--encoder-dropout-in', type=float, metavar='D',
                            help='dropout probability for encoder input embedding')
        parser.add_argument('--encoder-dropout-out', type=float, metavar='D',
                            help='dropout probability for encoder output')
        parser.add_argument('--decoder-dropout-in', type=float, metavar='D',
                            help='dropout probability for decoder input embedding')
        parser.add_argument('--decoder-dropout-out', type=float, metavar='D',
                            help='dropout probability for decoder output')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)
        base_architecture(args)

        if args.encoder_layers != args.decoder_layers:
            raise ValueError('--encoder-layers must match --decoder-layers')

        max_source_positions = getattr(args, 'max_source_positions', DEFAULT_MAX_SOURCE_POSITIONS)
        max_target_positions = getattr(args, 'max_target_positions', DEFAULT_MAX_TARGET_POSITIONS)

        def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
            embed_dict = utils.parse_embedding(embed_path)
            utils.print_embed_overlap(embed_dict, dictionary)
            return utils.load_embedding(embed_dict, dictionary, embed_tokens)

        if args.encoder_embed_path:
            pretrained_encoder_embed = load_pretrained_embedding_from_file(
                args.encoder_embed_path, task.source_dictionary, args.encoder_embed_dim)
        else:
            num_embeddings = len(task.source_dictionary)
            pretrained_encoder_embed = Embedding(
                num_embeddings, args.encoder_embed_dim, task.source_dictionary.pad()
            )

        if args.share_all_embeddings:
            # double check all parameters combinations are valid
            if task.source_dictionary != task.target_dictionary:
                raise ValueError('--share-all-embeddings requires a joint dictionary')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError(
                    '--share-all-embed not compatible with --decoder-embed-path'
                )
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to '
                    'match --decoder-embed-dim'
                )
            pretrained_decoder_embed = pretrained_encoder_embed
            args.share_decoder_input_output_embed = True
        else:
            # separate decoder input embeddings
            pretrained_decoder_embed = None
            if args.decoder_embed_path:
                pretrained_decoder_embed = load_pretrained_embedding_from_file(
                    args.decoder_embed_path,
                    task.target_dictionary,
                    args.decoder_embed_dim
                )
        # one last double check of parameter combinations
        if args.share_decoder_input_output_embed and (
                args.decoder_embed_dim != args.decoder_out_embed_dim):
            raise ValueError(
                '--share-decoder-input-output-embeddings requires '
                '--decoder-embed-dim to match --decoder-out-embed-dim'
            )

        if args.encoder_freeze_embed:
            pretrained_encoder_embed.weight.requires_grad = False
        if args.decoder_freeze_embed:
            pretrained_decoder_embed.weight.requires_grad = False

        encoder = LSTMEncoder(
            dictionary=task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            hidden_size=args.encoder_hidden_size,
            num_layers=args.encoder_layers,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
            bidirectional=args.encoder_bidirectional,
            pretrained_embed=pretrained_encoder_embed,
            max_source_positions=max_source_positions
        )
        g_encoder = None
        if args.with_amr:
            g_encoder = graph_encoder.GraphEncoder(dictionary=task.amr_dict,
                                                   embedding_dim=args.graph_encoder_embed_dim,
                                                   output_dim=args.decoder_hidden_size,
                                                   dropout=args.graph_in_dropout,
                                                   pad_idx=task.amr_dict.pad(),
                                                   n_layers=args.n_graph_layers,
                                                   aggr=args.aggr,
                                                   concat=args.concat_in_aggr,
                                                   n_highway=args.n_highways,
                                                   direction=args.direction,
                                                   )

        decoder = LSTMDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_hidden_size,
            with_graph=args.with_amr,
            graph_hidden_dim=args.graph_hidden_dim,
            concat_in_aggr=True if args.concat_in_aggr == "True" else False,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            attention=options.eval_bool(args.decoder_attention),
            encoder_output_units=encoder.output_units,
            pretrained_embed=pretrained_decoder_embed,
            share_input_output_embed=args.share_decoder_input_output_embed,
            adaptive_softmax_cutoff=(
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                if args.criterion == 'adaptive_loss' else None
            ),
            max_target_positions=max_target_positions
        )
        return cls(encoder, decoder, graph_encoder=g_encoder)


class LSTMEncoder(FairseqEncoder):
    """LSTM encoder."""

    def __init__(
            self, dictionary, embed_dim=512, hidden_size=512, num_layers=1,
            dropout_in=0.1, dropout_out=0.1, bidirectional=False,
            left_pad=True, pretrained_embed=None, padding_idx=None,
            max_source_positions=DEFAULT_MAX_SOURCE_POSITIONS
    ):
        super().__init__(dictionary)
        self.num_layers = num_layers
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.max_source_positions = max_source_positions

        num_embeddings = len(dictionary)
        self.padding_idx = padding_idx if padding_idx is not None else dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.lstm = LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.dropout_out if num_layers > 1 else 0.,
            bidirectional=bidirectional,
        )
        self.left_pad = left_pad

        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2

    def forward(self, src_tokens, src_lengths):
        if self.left_pad:
            # nn.utils.rnn.pack_padded_sequence requires right-padding;
            # convert left-padding to right-padding
            src_tokens = utils.convert_padding_direction(
                src_tokens,
                self.padding_idx,
                left_to_right=True,
            )

        bsz, seqlen = src_tokens.size()

        # embed tokens
        x = self.embed_tokens(src_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # pack embedded source tokens into a PackedSequence
        packed_x = nn.utils.rnn.pack_padded_sequence(x, src_lengths.data.tolist())

        # apply LSTM
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        h0 = x.new_zeros(*state_size)
        c0 = x.new_zeros(*state_size)
        packed_outs, (final_hiddens, final_cells) = self.lstm(packed_x, (h0, c0))

        # unpack outputs and apply dropout
        x, _ = nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=self.padding_idx)
        x = F.dropout(x, p=self.dropout_out, training=self.training)
        assert list(x.size()) == [seqlen, bsz, self.output_units]

        if self.bidirectional:
            def combine_bidir(outs):
                out = outs.view(self.num_layers, 2, bsz, -1).transpose(1, 2).contiguous()
                return out.view(self.num_layers, bsz, -1)

            final_hiddens = combine_bidir(final_hiddens)
            final_cells = combine_bidir(final_cells)

        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()

        return {
            'encoder_out': (x, final_hiddens, final_cells),
            'encoder_padding_mask': encoder_padding_mask if encoder_padding_mask.any() else None
        }

    def reorder_encoder_out(self, encoder_out, new_order, is_graph=False):
        index = 1 if is_graph is False else 0
        encoder_out['encoder_out'] = tuple(
            eo.index_select(index, new_order)
            for eo in encoder_out['encoder_out']
        )
        if encoder_out['encoder_padding_mask'] is not None:
            if is_graph:
                index += 1
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(index, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.max_source_positions


class AttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, source_embed_dim, output_embed_dim, bias=False):
        super().__init__()

        self.input_proj = Linear(input_embed_dim, source_embed_dim, bias=bias)
        self.output_proj = Linear(input_embed_dim + source_embed_dim, output_embed_dim, bias=bias)

    def forward(self, input, source_hids, encoder_padding_mask):
        # input: bsz x input_embed_dim
        # source_hids: srclen x bsz x source_embed_dim

        # x: bsz x source_embed_dim
        x = self.input_proj(input)
        # compute attention
        attn_scores = (source_hids * x.unsqueeze(0)).sum(dim=2)

        # don't attend over padding
        if encoder_padding_mask is not None:
            attn_scores = attn_scores.float().masked_fill_(
                encoder_padding_mask,
                float('-inf')
            ).type_as(attn_scores)  # FP16 support: cast to float and back

        attn_scores = F.softmax(attn_scores, dim=0)  # srclen x bsz

        # sum weighted sources
        x = (attn_scores.unsqueeze(2) * source_hids).sum(dim=0)

        x = torch.tanh(self.output_proj(torch.cat((x, input), dim=1)))
        return x, attn_scores


class LSTMDecoder(FairseqIncrementalDecoder):
    """LSTM decoder."""

    def __init__(
            self, dictionary, embed_dim=512, hidden_size=512, graph_hidden_dim=512, concat_in_aggr=True,
            out_embed_dim=512,
            num_layers=1, dropout_in=0.1, dropout_out=0.1, attention=True,
            encoder_output_units=512, pretrained_embed=None, with_graph=False,
            share_input_output_embed=False, adaptive_softmax_cutoff=None,
            max_target_positions=DEFAULT_MAX_TARGET_POSITIONS
    ):
        super().__init__(dictionary)
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.hidden_size = hidden_size
        self.share_input_output_embed = share_input_output_embed
        self.need_attn = True
        self.max_target_positions = max_target_positions
        self.with_graph = with_graph
        self.adaptive_softmax = None
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.encoder_output_units = encoder_output_units
        if encoder_output_units != hidden_size and encoder_output_units != 0:
            self.encoder_hidden_proj = Linear(encoder_output_units, hidden_size)
            self.encoder_cell_proj = Linear(encoder_output_units, hidden_size)
        else:
            self.encoder_hidden_proj = self.encoder_cell_proj = None

        # if with_graph:
        #     graph_hidden_dim = graph_hidden_dim * 4 if concat_in_aggr else graph_hidden_dim * 2
        #     if graph_hidden_dim != hidden_size:
        #         self.g_encoder_hidden_proj = Linear(graph_hidden_dim, hidden_size)
        #         self.g_encoder_output_proj = Linear(graph_hidden_dim, hidden_size)
        #     else:
        #         self.g_encoder_output_proj = self.g_encoder_hidden_proj = None

        # disable input feeding if there is no encoder
        # input feeding is described in arxiv.org/abs/1508.04025
        input_feed_size = 0 if encoder_output_units == 0 else hidden_size
        if with_graph:
            # graph_hidden_dim = graph_hidden_dim * 4 if concat_in_aggr else graph_hidden_dim * 2
            self.layers = nn.ModuleList([LSTMCell(
                input_size=input_feed_size + embed_dim + embed_dim if layer == 0 else hidden_size + graph_hidden_dim,
                hidden_size=hidden_size) for layer in range(num_layers)])
        else:
            self.layers = nn.ModuleList([
                LSTMCell(
                    input_size=input_feed_size + embed_dim if layer == 0 else hidden_size,
                    hidden_size=hidden_size,
                )
                for layer in range(num_layers)
            ])
        if attention:
            # TODO make bias configurable
            self.src_attention = AttentionLayer(hidden_size, encoder_output_units, hidden_size, bias=False)
            if self.with_graph:
                self.graph_attention = AttentionLayer(hidden_size, hidden_size, hidden_size,
                                                      bias=False)
        else:
            self.src_attention = None
            self.graph_attention = None
        if hidden_size != out_embed_dim:
            self.additional_fc = Linear(hidden_size, out_embed_dim)
        if adaptive_softmax_cutoff is not None:
            # setting adaptive_softmax dropout to dropout_out for now but can be redefined
            self.adaptive_softmax = AdaptiveSoftmax(num_embeddings, hidden_size, adaptive_softmax_cutoff,
                                                    dropout=dropout_out)
        elif not self.share_input_output_embed:
            self.fc_out = Linear(out_embed_dim, num_embeddings, dropout=dropout_out)

    def forward(self, prev_output_tokens, encoder_out=None, graph_encoder_out=None, incremental_state=None, **kwargs):
        x, attn_scores = self.extract_features(
            prev_output_tokens, encoder_out, graph_encoder_out, incremental_state
        )
        return self.output_layer(x), attn_scores

    def extract_features(
            self, prev_output_tokens, encoder_out, graph_encoder_out=None, incremental_state=None
    ):
        """
        Similar to *forward* but only return features.
        """
        # TODO: change decoder initial states into avg of graph hidden states
        # TODO: add highway layers as bridges in graph encoder
        if encoder_out is not None:
            encoder_padding_mask = encoder_out['encoder_padding_mask']
            encoder_out = encoder_out['encoder_out']
        else:
            encoder_padding_mask = None
            encoder_out = None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
        bsz, seqlen = prev_output_tokens.size()

        # get outputs from encoder
        if encoder_out is not None:
            encoder_outs, encoder_hiddens, encoder_cells = encoder_out[:3]
            srclen = encoder_outs.size(0)
        else:
            srclen = None

        if graph_encoder_out is not None:
            graph_encoder_outs, node_mask = graph_encoder_out.values()
            graph_hidden, graph_encoder_output = graph_encoder_outs[:2]
            # node_mask, in_indice_mask, in_edge_mask, out_indice_mask, out_edge_mask = graph_mask.values()

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # initialize previous states (or get from cache during incremental generation)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is not None:
            prev_hiddens, prev_cells, input_feed, graph_input_feed = cached_state
        elif encoder_out is not None:
            # setup recurrent cells
            num_layers = len(self.layers)
            prev_hiddens = [encoder_hiddens[i] for i in range(num_layers)]
            prev_cells = [encoder_cells[i] for i in range(num_layers)]
            if self.encoder_hidden_proj is not None:
                prev_hiddens = [self.encoder_hidden_proj(x) for x in prev_hiddens]
                prev_cells = [self.encoder_cell_proj(x) for x in prev_cells]
            input_feed = x.new_zeros(bsz, self.hidden_size)
            graph_input_feed = None
            if graph_encoder_out is not None:
                graph_input_feed = x.new_zeros(bsz, self.hidden_size)
                # if self.g_encoder_hidden_proj is not None:
                #     graph_hidden = self.g_encoder_hidden_proj(graph_hidden)
                #     graph_encoder_output = self.g_encoder_output_proj(graph_encoder_output)
        else:
            # setup zero cells, since there is no encoder
            num_layers = len(self.layers)
            zero_state = x.new_zeros(bsz, self.hidden_size)
            prev_hiddens = [zero_state for i in range(num_layers)]
            prev_cells = [zero_state for i in range(num_layers)]
            input_feed = None
            graph_input_feed = None

        assert srclen is not None or self.src_attention is None, \
            "attention is not supported if there are no encoder outputs"
        attn_scores = x.new_zeros(srclen, seqlen, bsz) if self.src_attention is not None else None
        # graph_attn_scores = x.new_zeros(graphlen, nodelen, bsz) if self.graph_attention is not None else None

        outs = []
        # gouts = []
        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            if input_feed is not None and graph_input_feed is not None:
                input = torch.cat((x[j, :, :], input_feed, graph_input_feed), dim=1)
            elif input_feed is not None and graph_input_feed is None:
                input = torch.cat((x[j, :, :], input_feed), dim=1)
            else:
                input = x[j]

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                input = F.dropout(hidden, p=self.dropout_out, training=self.training)

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            # apply attention using the last layer's hidden state
            if self.src_attention is not None:
                src_out, attn_scores[:, j, :] = self.src_attention(hidden, encoder_outs, encoder_padding_mask)
                if self.with_graph:
                    graph_out, _ = self.graph_attention(hidden, graph_encoder_output.permute(1, 0, 2), node_mask)
            else:
                src_out = hidden
                graph_out = hidden
            src_out = F.dropout(src_out, p=self.dropout_out, training=self.training)
            if self.with_graph:
                graph_out = F.dropout(graph_out, p=self.dropout_out, training=self.training)
            # input feeding
            if input_feed is not None:
                input_feed = src_out
            if graph_input_feed is not None:
                graph_input_feed = graph_out
            # save final output
            if self.with_graph:
                outs.append(src_out + graph_out)
            else:
                outs.append(src_out)
            # gouts.append(graph_out)
        # cache previous states (no-op except during incremental generation)
        utils.set_incremental_state(
            self, incremental_state, 'cached_state',
            (prev_hiddens, prev_cells, input_feed, graph_input_feed),
        )

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hidden_size)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        if hasattr(self, 'additional_fc') and self.adaptive_softmax is None:
            x = self.additional_fc(x)
            x = F.dropout(x, p=self.dropout_out, training=self.training)

        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        if not self.training and self.need_attn and self.src_attention is not None:
            attn_scores = attn_scores.transpose(0, 2)
        else:
            attn_scores = None
        return x, attn_scores

    def output_layer(self, x):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = self.fc_out(x)
        return x

    def reorder_incremental_state(self, incremental_state, new_order):
        super().reorder_incremental_state(incremental_state, new_order)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is None:
            return

        def reorder_state(state):
            if isinstance(state, list):
                return [reorder_state(state_i) for state_i in state]
            elif state is not None:
                return state.index_select(0, new_order)
            else:
                return None

        new_state = tuple(map(reorder_state, cached_state))
        utils.set_incremental_state(self, incremental_state, 'cached_state', new_state)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.max_target_positions

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.uniform_(m.weight, -0.1, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def LSTM(input_size, hidden_size, **kwargs):
    m = nn.LSTM(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def LSTMCell(input_size, hidden_size, **kwargs):
    m = nn.LSTMCell(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def Linear(in_features, out_features, bias=True, dropout=0):
    """Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m


@register_model_architecture('lstm', 'lstm')
def base_architecture(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_freeze_embed = getattr(args, 'encoder_freeze_embed', False)
    args.encoder_hidden_size = getattr(args, 'encoder_hidden_size', args.encoder_embed_dim)
    args.encoder_layers = getattr(args, 'encoder_layers', 1)
    args.encoder_compress_dim = getattr(args, 'encoder_compress_dim', 512)
    args.encoder_bidirectional = getattr(args, 'encoder_bidirectional', False)
    args.encoder_dropout_in = getattr(args, 'encoder_dropout_in', args.dropout)
    args.encoder_dropout_out = getattr(args, 'encoder_dropout_out', args.dropout)

    args.graph_encoder_embed_dim = getattr(args, 'graph_encoder_embed_dim', 128)
    args.graph_hidden_dim = 2 * args.graph_encoder_embed_dim
    args.aggr = getattr(args, 'aggr', "maxpooling")
    args.graph_in_dropout = getattr(args, 'graph_in_dropout', 0.2)
    args.graph_out_dropout = getattr(args, 'graph_out_dropout', 0.2)
    args.n_graph_layers = getattr(args, 'n_graph_layers', 2)
    args.concat_in_aggr = getattr(args, 'concat_in_aggr', True)
    args.n_highways = getattr(args, 'n_highways', 1)
    args.direction = getattr(args, 'direction', 'bi')

    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_freeze_embed = getattr(args, 'decoder_freeze_embed', False)
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', args.decoder_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 1)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 512)
    args.decoder_attention = getattr(args, 'decoder_attention', '1')
    args.decoder_dropout_in = getattr(args, 'decoder_dropout_in', args.dropout)
    args.decoder_dropout_out = getattr(args, 'decoder_dropout_out', args.dropout)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', '10000,50000,200000')


@register_model_architecture('lstm', 'lstm_wiseman_iwslt_de_en')
def lstm_wiseman_iwslt_de_en(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_dropout_in = getattr(args, 'encoder_dropout_in', 0)
    args.encoder_dropout_out = getattr(args, 'encoder_dropout_out', 0)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 256)
    args.decoder_dropout_in = getattr(args, 'decoder_dropout_in', 0)
    args.decoder_dropout_out = getattr(args, 'decoder_dropout_out', args.dropout)
    base_architecture(args)


@register_model_architecture('lstm', 'lstm_luong_wmt_en_de')
def lstm_luong_wmt_en_de(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1000)
    args.encoder_layers = getattr(args, 'encoder_layers', 4)
    args.encoder_dropout_out = getattr(args, 'encoder_dropout_out', 0)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1000)
    args.decoder_layers = getattr(args, 'decoder_layers', 4)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 1000)
    args.decoder_dropout_out = getattr(args, 'decoder_dropout_out', 0)
    base_architecture(args)
