
Á
B
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
:
FloorMod
x"T
y"T
z"T"
Ttype:
	2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
:
Maximum
x"T
y"T
z"T"
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
:
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
¥
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
-
Sqrt
x"T
y"T"
Ttype:

2
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8
z
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_9/kernel
s
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel* 
_output_shapes
:
*
dtype0
q
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_9/bias
j
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes	
:*
dtype0
|
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense_10/kernel
u
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel* 
_output_shapes
:
*
dtype0
s
dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_10/bias
l
!dense_10/bias/Read/ReadVariableOpReadVariableOpdense_10/bias*
_output_shapes	
:*
dtype0
{
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	* 
shared_namedense_12/kernel
t
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel*
_output_shapes
:	*
dtype0
s
dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_12/bias
l
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias*
_output_shapes	
:*
dtype0
|
dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense_11/kernel
u
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel* 
_output_shapes
:
*
dtype0
s
dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_11/bias
l
!dense_11/bias/Read/ReadVariableOpReadVariableOpdense_11/bias*
_output_shapes	
:*
dtype0
|
dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense_13/kernel
u
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel* 
_output_shapes
:
*
dtype0
s
dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_13/bias
l
!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias*
_output_shapes	
:*
dtype0
|
dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense_14/kernel
u
#dense_14/kernel/Read/ReadVariableOpReadVariableOpdense_14/kernel* 
_output_shapes
:
*
dtype0
s
dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_14/bias
l
!dense_14/bias/Read/ReadVariableOpReadVariableOpdense_14/bias*
_output_shapes	
:*
dtype0
|
dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense_15/kernel
u
#dense_15/kernel/Read/ReadVariableOpReadVariableOpdense_15/kernel* 
_output_shapes
:
*
dtype0
s
dense_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_15/bias
l
!dense_15/bias/Read/ReadVariableOpReadVariableOpdense_15/bias*
_output_shapes	
:*
dtype0
|
dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense_16/kernel
u
#dense_16/kernel/Read/ReadVariableOpReadVariableOpdense_16/kernel* 
_output_shapes
:
*
dtype0
s
dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_16/bias
l
!dense_16/bias/Read/ReadVariableOpReadVariableOpdense_16/bias*
_output_shapes	
:*
dtype0
{
dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	* 
shared_namedense_17/kernel
t
#dense_17/kernel/Read/ReadVariableOpReadVariableOpdense_17/kernel*
_output_shapes
:	*
dtype0
r
dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_17/bias
k
!dense_17/bias/Read/ReadVariableOpReadVariableOpdense_17/bias*
_output_shapes
:*
dtype0

embedding_8/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	4*'
shared_nameembedding_8/embeddings

*embedding_8/embeddings/Read/ReadVariableOpReadVariableOpembedding_8/embeddings*
_output_shapes
:	4*
dtype0

embedding_6/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameembedding_6/embeddings

*embedding_6/embeddings/Read/ReadVariableOpReadVariableOpembedding_6/embeddings*
_output_shapes
:	*
dtype0

embedding_7/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameembedding_7/embeddings

*embedding_7/embeddings/Read/ReadVariableOpReadVariableOpembedding_7/embeddings*
_output_shapes
:	*
dtype0

embedding_11/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	4*(
shared_nameembedding_11/embeddings

+embedding_11/embeddings/Read/ReadVariableOpReadVariableOpembedding_11/embeddings*
_output_shapes
:	4*
dtype0

embedding_9/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameembedding_9/embeddings

*embedding_9/embeddings/Read/ReadVariableOpReadVariableOpembedding_9/embeddings*
_output_shapes
:	*
dtype0

embedding_10/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*(
shared_nameembedding_10/embeddings

+embedding_10/embeddings/Read/ReadVariableOpReadVariableOpembedding_10/embeddings*
_output_shapes
:	*
dtype0

 normalize_1/normalization_1/meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" normalize_1/normalization_1/mean

4normalize_1/normalization_1/mean/Read/ReadVariableOpReadVariableOp normalize_1/normalization_1/mean*
_output_shapes	
:*
dtype0
¡
$normalize_1/normalization_1/varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$normalize_1/normalization_1/variance

8normalize_1/normalization_1/variance/Read/ReadVariableOpReadVariableOp$normalize_1/normalization_1/variance*
_output_shapes	
:*
dtype0

!normalize_1/normalization_1/countVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *2
shared_name#!normalize_1/normalization_1/count

5normalize_1/normalization_1/count/Read/ReadVariableOpReadVariableOp!normalize_1/normalization_1/count*
_output_shapes
: *
dtype0	
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *    
L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 * $tI
L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *    

NoOpNoOp
Øa
Const_5Const"/device:CPU:0*
_output_shapes
: *
dtype0*a
valueaBa Bý`
þ
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-2

layer-9
layer-10
layer_with_weights-3
layer-11
layer_with_weights-4
layer-12
layer_with_weights-5
layer-13
layer_with_weights-6
layer-14
layer-15
layer_with_weights-7
layer-16
layer-17
layer_with_weights-8
layer-18
layer-19
layer-20
layer_with_weights-9
layer-21
layer-22
layer-23
layer_with_weights-10
layer-24
layer_with_weights-11
layer-25
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
 
 
è
 layer-0
!layer-1
"layer-2
#layer-3
$layer_with_weights-0
$layer-4
%layer_with_weights-1
%layer-5
&layer-6
'layer-7
(layer-8
)layer_with_weights-2
)layer-9
*layer-10
+layer-11
,layer-12
-layer-13
.layer-14
/trainable_variables
0regularization_losses
1	variables
2	keras_api
è
3layer-0
4layer-1
5layer-2
6layer-3
7layer_with_weights-0
7layer-4
8layer_with_weights-1
8layer-5
9layer-6
:layer-7
;layer-8
<layer_with_weights-2
<layer-9
=layer-10
>layer-11
?layer-12
@layer-13
Alayer-14
Btrainable_variables
Cregularization_losses
D	variables
E	keras_api

F	keras_api

G	keras_api

H	keras_api

I	keras_api
h

Jkernel
Kbias
Ltrainable_variables
Mregularization_losses
N	variables
O	keras_api

P	keras_api
h

Qkernel
Rbias
Strainable_variables
Tregularization_losses
U	variables
V	keras_api
h

Wkernel
Xbias
Ytrainable_variables
Zregularization_losses
[	variables
\	keras_api
h

]kernel
^bias
_trainable_variables
`regularization_losses
a	variables
b	keras_api
h

ckernel
dbias
etrainable_variables
fregularization_losses
g	variables
h	keras_api

i	keras_api
h

jkernel
kbias
ltrainable_variables
mregularization_losses
n	variables
o	keras_api

p	keras_api
h

qkernel
rbias
strainable_variables
tregularization_losses
u	variables
v	keras_api

w	keras_api

x	keras_api
h

ykernel
zbias
{trainable_variables
|regularization_losses
}	variables
~	keras_api

	keras_api

	keras_api
f
	normalize
trainable_variables
regularization_losses
	variables
	keras_api
n
kernel
	bias
trainable_variables
regularization_losses
	variables
	keras_api
¾
0
1
2
3
4
5
J6
K7
Q8
R9
W10
X11
]12
^13
c14
d15
j16
k17
q18
r19
y20
z21
22
23
 
Ù
0
1
2
3
4
5
J6
K7
Q8
R9
W10
X11
]12
^13
c14
d15
j16
k17
q18
r19
y20
z21
22
23
24
25
26
²
layers
metrics
 layer_regularization_losses
trainable_variables
layer_metrics
regularization_losses
non_trainable_variables
	variables
 
 
V
trainable_variables
regularization_losses
	variables
	keras_api

	keras_api

	keras_api
g

embeddings
 trainable_variables
¡regularization_losses
¢	variables
£	keras_api
g

embeddings
¤trainable_variables
¥regularization_losses
¦	variables
§	keras_api

¨	keras_api

©	keras_api

ª	keras_api
g

embeddings
«trainable_variables
¬regularization_losses
­	variables
®	keras_api

¯	keras_api

°	keras_api

±	keras_api

²	keras_api

³	keras_api

0
1
2
 

0
1
2
²
´layers
µmetrics
 ¶layer_regularization_losses
/trainable_variables
·layer_metrics
0regularization_losses
¸non_trainable_variables
1	variables
 
V
¹trainable_variables
ºregularization_losses
»	variables
¼	keras_api

½	keras_api

¾	keras_api
g

embeddings
¿trainable_variables
Àregularization_losses
Á	variables
Â	keras_api
g

embeddings
Ãtrainable_variables
Äregularization_losses
Å	variables
Æ	keras_api

Ç	keras_api

È	keras_api

É	keras_api
g

embeddings
Êtrainable_variables
Ëregularization_losses
Ì	variables
Í	keras_api

Î	keras_api

Ï	keras_api

Ð	keras_api

Ñ	keras_api

Ò	keras_api

0
1
2
 

0
1
2
²
Ólayers
Ômetrics
 Õlayer_regularization_losses
Btrainable_variables
Ölayer_metrics
Cregularization_losses
×non_trainable_variables
D	variables
 
 
 
 
ZX
VARIABLE_VALUEdense_9/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_9/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

J0
K1
 

J0
K1
²
Ølayers
Ùmetrics
 Úlayer_regularization_losses
Ltrainable_variables
Ûlayer_metrics
Mregularization_losses
Ünon_trainable_variables
N	variables
 
[Y
VARIABLE_VALUEdense_10/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_10/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

Q0
R1
 

Q0
R1
²
Ýlayers
Þmetrics
 ßlayer_regularization_losses
Strainable_variables
àlayer_metrics
Tregularization_losses
ánon_trainable_variables
U	variables
[Y
VARIABLE_VALUEdense_12/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_12/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

W0
X1
 

W0
X1
²
âlayers
ãmetrics
 älayer_regularization_losses
Ytrainable_variables
ålayer_metrics
Zregularization_losses
ænon_trainable_variables
[	variables
[Y
VARIABLE_VALUEdense_11/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_11/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

]0
^1
 

]0
^1
²
çlayers
èmetrics
 élayer_regularization_losses
_trainable_variables
êlayer_metrics
`regularization_losses
ënon_trainable_variables
a	variables
[Y
VARIABLE_VALUEdense_13/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_13/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

c0
d1
 

c0
d1
²
ìlayers
ímetrics
 îlayer_regularization_losses
etrainable_variables
ïlayer_metrics
fregularization_losses
ðnon_trainable_variables
g	variables
 
[Y
VARIABLE_VALUEdense_14/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_14/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

j0
k1
 

j0
k1
²
ñlayers
òmetrics
 ólayer_regularization_losses
ltrainable_variables
ôlayer_metrics
mregularization_losses
õnon_trainable_variables
n	variables
 
[Y
VARIABLE_VALUEdense_15/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_15/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

q0
r1
 

q0
r1
²
ölayers
÷metrics
 ølayer_regularization_losses
strainable_variables
ùlayer_metrics
tregularization_losses
únon_trainable_variables
u	variables
 
 
[Y
VARIABLE_VALUEdense_16/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_16/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

y0
z1
 

y0
z1
²
ûlayers
ümetrics
 ýlayer_regularization_losses
{trainable_variables
þlayer_metrics
|regularization_losses
ÿnon_trainable_variables
}	variables
 
 
c
state_variables
_broadcast_shape
	mean
variance

count
	keras_api
 
 

0
1
2
µ
layers
metrics
 layer_regularization_losses
trainable_variables
layer_metrics
regularization_losses
non_trainable_variables
	variables
\Z
VARIABLE_VALUEdense_17/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_17/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
µ
layers
metrics
 layer_regularization_losses
trainable_variables
layer_metrics
regularization_losses
non_trainable_variables
	variables
\Z
VARIABLE_VALUEembedding_8/embeddings0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEembedding_6/embeddings0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEembedding_7/embeddings0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEembedding_11/embeddings0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEembedding_9/embeddings0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEembedding_10/embeddings0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE normalize_1/normalization_1/mean'variables/22/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE$normalize_1/normalization_1/variance'variables/23/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!normalize_1/normalization_1/count'variables/24/.ATTRIBUTES/VARIABLE_VALUE
Æ
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25

0
 
 

0
1
2
 
 
 
µ
layers
metrics
 layer_regularization_losses
trainable_variables
layer_metrics
regularization_losses
non_trainable_variables
	variables
 
 

0
 

0
µ
layers
metrics
 layer_regularization_losses
 trainable_variables
layer_metrics
¡regularization_losses
non_trainable_variables
¢	variables

0
 

0
µ
layers
metrics
 layer_regularization_losses
¤trainable_variables
layer_metrics
¥regularization_losses
non_trainable_variables
¦	variables
 
 
 

0
 

0
µ
layers
metrics
 layer_regularization_losses
«trainable_variables
 layer_metrics
¬regularization_losses
¡non_trainable_variables
­	variables
 
 
 
 
 
n
 0
!1
"2
#3
$4
%5
&6
'7
(8
)9
*10
+11
,12
-13
.14
 
 
 
 
 
 
 
µ
¢layers
£metrics
 ¤layer_regularization_losses
¹trainable_variables
¥layer_metrics
ºregularization_losses
¦non_trainable_variables
»	variables
 
 

0
 

0
µ
§layers
¨metrics
 ©layer_regularization_losses
¿trainable_variables
ªlayer_metrics
Àregularization_losses
«non_trainable_variables
Á	variables

0
 

0
µ
¬layers
­metrics
 ®layer_regularization_losses
Ãtrainable_variables
¯layer_metrics
Äregularization_losses
°non_trainable_variables
Å	variables
 
 
 

0
 

0
µ
±layers
²metrics
 ³layer_regularization_losses
Êtrainable_variables
´layer_metrics
Ëregularization_losses
µnon_trainable_variables
Ì	variables
 
 
 
 
 
n
30
41
52
63
74
85
96
:7
;8
<9
=10
>11
?12
@13
A14
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
&
	mean
variance

count
 
 

0
 
 
 

0
1
2
 
 
 
 
 
8

¶total

·count
¸	variables
¹	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

¶0
·1

¸	variables
w
serving_default_betsPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ

y
serving_default_cards0Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
y
serving_default_cards1Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
À
StatefulPartitionedCallStatefulPartitionedCallserving_default_betsserving_default_cards0serving_default_cards1ConstConst_1embedding_8/embeddingsembedding_6/embeddingsembedding_7/embeddingsConst_2embedding_11/embeddingsembedding_9/embeddingsembedding_10/embeddingsConst_3Const_4dense_9/kerneldense_9/biasdense_12/kerneldense_12/biasdense_10/kerneldense_10/biasdense_11/kerneldense_11/biasdense_13/kerneldense_13/biasdense_14/kerneldense_14/biasdense_15/kerneldense_15/biasdense_16/kerneldense_16/bias normalize_1/normalization_1/mean$normalize_1/normalization_1/variancedense_17/kerneldense_17/bias*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
 !*2
config_proto" 

CPU

GPU2 *0J 8 *+
f&R$
"__inference_signature_wrapper_5580
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ª
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOp#dense_10/kernel/Read/ReadVariableOp!dense_10/bias/Read/ReadVariableOp#dense_12/kernel/Read/ReadVariableOp!dense_12/bias/Read/ReadVariableOp#dense_11/kernel/Read/ReadVariableOp!dense_11/bias/Read/ReadVariableOp#dense_13/kernel/Read/ReadVariableOp!dense_13/bias/Read/ReadVariableOp#dense_14/kernel/Read/ReadVariableOp!dense_14/bias/Read/ReadVariableOp#dense_15/kernel/Read/ReadVariableOp!dense_15/bias/Read/ReadVariableOp#dense_16/kernel/Read/ReadVariableOp!dense_16/bias/Read/ReadVariableOp#dense_17/kernel/Read/ReadVariableOp!dense_17/bias/Read/ReadVariableOp*embedding_8/embeddings/Read/ReadVariableOp*embedding_6/embeddings/Read/ReadVariableOp*embedding_7/embeddings/Read/ReadVariableOp+embedding_11/embeddings/Read/ReadVariableOp*embedding_9/embeddings/Read/ReadVariableOp+embedding_10/embeddings/Read/ReadVariableOp4normalize_1/normalization_1/mean/Read/ReadVariableOp8normalize_1/normalization_1/variance/Read/ReadVariableOp5normalize_1/normalization_1/count/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst_5**
Tin#
!2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *&
f!R
__inference__traced_save_6719
ß
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_9/kerneldense_9/biasdense_10/kerneldense_10/biasdense_12/kerneldense_12/biasdense_11/kerneldense_11/biasdense_13/kerneldense_13/biasdense_14/kerneldense_14/biasdense_15/kerneldense_15/biasdense_16/kerneldense_16/biasdense_17/kerneldense_17/biasembedding_8/embeddingsembedding_6/embeddingsembedding_7/embeddingsembedding_11/embeddingsembedding_9/embeddingsembedding_10/embeddings normalize_1/normalization_1/mean$normalize_1/normalization_1/variance!normalize_1/normalization_1/counttotalcount*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *)
f$R"
 __inference__traced_restore_6816øã
á
|
'__inference_dense_10_layer_call_fn_6318

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_dense_10_layer_call_and_return_conditional_losses_48802
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
©
_
C__inference_flatten_3_layer_call_and_return_conditional_losses_6546

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù	

E__inference_embedding_9_layer_call_and_return_conditional_losses_6578

inputs
embedding_lookup_6572
identity¢embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Castú
embedding_lookupResourceGatherembedding_lookup_6572Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*(
_class
loc:@embedding_lookup/6572*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_lookupì
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@embedding_lookup/6572*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity¡
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼

&__inference_model_2_layer_call_fn_4458
input_3
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_model_2_layer_call_and_return_conditional_losses_44472
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: :::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_3:

_output_shapes
: 
©
_
C__inference_flatten_2_layer_call_and_return_conditional_losses_6484

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ß
{
&__inference_dense_9_layer_call_fn_6298

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_dense_9_layer_call_and_return_conditional_losses_48272
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
-
À
A__inference_model_2_layer_call_and_return_conditional_losses_4492

inputs*
&tf_math_greater_equal_3_greaterequal_y
embedding_8_4474
embedding_6_4477
embedding_7_4482
identity¢#embedding_6/StatefulPartitionedCall¢#embedding_7/StatefulPartitionedCall¢#embedding_8/StatefulPartitionedCallÚ
flatten_2/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_flatten_2_layer_call_and_return_conditional_losses_42872
flatten_2/PartitionedCall
*tf.clip_by_value_3/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_3/clip_by_value/Minimum/yê
(tf.clip_by_value_3/clip_by_value/MinimumMinimum"flatten_2/PartitionedCall:output:03tf.clip_by_value_3/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(tf.clip_by_value_3/clip_by_value/Minimum
"tf.clip_by_value_3/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_3/clip_by_value/yÜ
 tf.clip_by_value_3/clip_by_valueMaximum,tf.clip_by_value_3/clip_by_value/Minimum:z:0+tf.clip_by_value_3/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 tf.clip_by_value_3/clip_by_value
#tf.compat.v1.floor_div_2/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2%
#tf.compat.v1.floor_div_2/FloorDiv/yØ
!tf.compat.v1.floor_div_2/FloorDivFloorDiv$tf.clip_by_value_3/clip_by_value:z:0,tf.compat.v1.floor_div_2/FloorDiv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!tf.compat.v1.floor_div_2/FloorDivÚ
$tf.math.greater_equal_3/GreaterEqualGreaterEqual"flatten_2/PartitionedCall:output:0&tf_math_greater_equal_3_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$tf.math.greater_equal_3/GreaterEqual
tf.math.floormod_2/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2
tf.math.floormod_2/FloorMod/yÆ
tf.math.floormod_2/FloorModFloorMod$tf.clip_by_value_3/clip_by_value:z:0&tf.math.floormod_2/FloorMod/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.floormod_2/FloorMod±
#embedding_8/StatefulPartitionedCallStatefulPartitionedCall$tf.clip_by_value_3/clip_by_value:z:0embedding_8_4474*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_embedding_8_layer_call_and_return_conditional_losses_43152%
#embedding_8/StatefulPartitionedCall²
#embedding_6/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_2/FloorDiv:z:0embedding_6_4477*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_embedding_6_layer_call_and_return_conditional_losses_43372%
#embedding_6/StatefulPartitionedCall
tf.cast_3/CastCast(tf.math.greater_equal_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.cast_3/CastØ
tf.__operators__.add_6/AddV2AddV2,embedding_8/StatefulPartitionedCall:output:0,embedding_6/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_6/AddV2¬
#embedding_7/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_2/FloorMod:z:0embedding_7_4482*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_embedding_7_layer_call_and_return_conditional_losses_43612%
#embedding_7/StatefulPartitionedCallÌ
tf.__operators__.add_7/AddV2AddV2 tf.__operators__.add_6/AddV2:z:0,embedding_7/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_7/AddV2
tf.expand_dims_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
tf.expand_dims_2/ExpandDims/dim¼
tf.expand_dims_2/ExpandDims
ExpandDimstf.cast_3/Cast:y:0(tf.expand_dims_2/ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.expand_dims_2/ExpandDims¶
tf.math.multiply_2/MulMul tf.__operators__.add_7/AddV2:z:0$tf.expand_dims_2/ExpandDims:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.multiply_2/Mul
*tf.math.reduce_sum_2/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_2/Sum/reduction_indices¿
tf.math.reduce_sum_2/SumSumtf.math.multiply_2/Mul:z:03tf.math.reduce_sum_2/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.reduce_sum_2/Sumè
IdentityIdentity!tf.math.reduce_sum_2/Sum:output:0$^embedding_6/StatefulPartitionedCall$^embedding_7/StatefulPartitionedCall$^embedding_8/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: :::2J
#embedding_6/StatefulPartitionedCall#embedding_6/StatefulPartitionedCall2J
#embedding_7/StatefulPartitionedCall#embedding_7/StatefulPartitionedCall2J
#embedding_8/StatefulPartitionedCall#embedding_8/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 

D
(__inference_flatten_3_layer_call_fn_6551

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_flatten_3_layer_call_and_return_conditional_losses_45132
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
8
Þ
A__inference_model_2_layer_call_and_return_conditional_losses_6142

inputs*
&tf_math_greater_equal_3_greaterequal_y%
!embedding_8_embedding_lookup_6116%
!embedding_6_embedding_lookup_6122%
!embedding_7_embedding_lookup_6130
identity¢embedding_6/embedding_lookup¢embedding_7/embedding_lookup¢embedding_8/embedding_lookups
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten_2/Const
flatten_2/ReshapeReshapeinputsflatten_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten_2/Reshape
*tf.clip_by_value_3/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_3/clip_by_value/Minimum/yâ
(tf.clip_by_value_3/clip_by_value/MinimumMinimumflatten_2/Reshape:output:03tf.clip_by_value_3/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(tf.clip_by_value_3/clip_by_value/Minimum
"tf.clip_by_value_3/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_3/clip_by_value/yÜ
 tf.clip_by_value_3/clip_by_valueMaximum,tf.clip_by_value_3/clip_by_value/Minimum:z:0+tf.clip_by_value_3/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 tf.clip_by_value_3/clip_by_value
#tf.compat.v1.floor_div_2/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2%
#tf.compat.v1.floor_div_2/FloorDiv/yØ
!tf.compat.v1.floor_div_2/FloorDivFloorDiv$tf.clip_by_value_3/clip_by_value:z:0,tf.compat.v1.floor_div_2/FloorDiv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!tf.compat.v1.floor_div_2/FloorDivÒ
$tf.math.greater_equal_3/GreaterEqualGreaterEqualflatten_2/Reshape:output:0&tf_math_greater_equal_3_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$tf.math.greater_equal_3/GreaterEqual
tf.math.floormod_2/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2
tf.math.floormod_2/FloorMod/yÆ
tf.math.floormod_2/FloorModFloorMod$tf.clip_by_value_3/clip_by_value:z:0&tf.math.floormod_2/FloorMod/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.floormod_2/FloorMod
embedding_8/CastCast$tf.clip_by_value_3/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_8/Cast¶
embedding_8/embedding_lookupResourceGather!embedding_8_embedding_lookup_6116embedding_8/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*4
_class*
(&loc:@embedding_8/embedding_lookup/6116*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_8/embedding_lookup
%embedding_8/embedding_lookup/IdentityIdentity%embedding_8/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*4
_class*
(&loc:@embedding_8/embedding_lookup/6116*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%embedding_8/embedding_lookup/IdentityÅ
'embedding_8/embedding_lookup/Identity_1Identity.embedding_8/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'embedding_8/embedding_lookup/Identity_1
embedding_6/CastCast%tf.compat.v1.floor_div_2/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_6/Cast¶
embedding_6/embedding_lookupResourceGather!embedding_6_embedding_lookup_6122embedding_6/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*4
_class*
(&loc:@embedding_6/embedding_lookup/6122*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_6/embedding_lookup
%embedding_6/embedding_lookup/IdentityIdentity%embedding_6/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*4
_class*
(&loc:@embedding_6/embedding_lookup/6122*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%embedding_6/embedding_lookup/IdentityÅ
'embedding_6/embedding_lookup/Identity_1Identity.embedding_6/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'embedding_6/embedding_lookup/Identity_1
tf.cast_3/CastCast(tf.math.greater_equal_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.cast_3/Castà
tf.__operators__.add_6/AddV2AddV20embedding_8/embedding_lookup/Identity_1:output:00embedding_6/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_6/AddV2
embedding_7/CastCasttf.math.floormod_2/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_7/Cast¶
embedding_7/embedding_lookupResourceGather!embedding_7_embedding_lookup_6130embedding_7/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*4
_class*
(&loc:@embedding_7/embedding_lookup/6130*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_7/embedding_lookup
%embedding_7/embedding_lookup/IdentityIdentity%embedding_7/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*4
_class*
(&loc:@embedding_7/embedding_lookup/6130*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%embedding_7/embedding_lookup/IdentityÅ
'embedding_7/embedding_lookup/Identity_1Identity.embedding_7/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'embedding_7/embedding_lookup/Identity_1Ð
tf.__operators__.add_7/AddV2AddV2 tf.__operators__.add_6/AddV2:z:00embedding_7/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_7/AddV2
tf.expand_dims_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
tf.expand_dims_2/ExpandDims/dim¼
tf.expand_dims_2/ExpandDims
ExpandDimstf.cast_3/Cast:y:0(tf.expand_dims_2/ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.expand_dims_2/ExpandDims¶
tf.math.multiply_2/MulMul tf.__operators__.add_7/AddV2:z:0$tf.expand_dims_2/ExpandDims:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.multiply_2/Mul
*tf.math.reduce_sum_2/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_2/Sum/reduction_indices¿
tf.math.reduce_sum_2/SumSumtf.math.multiply_2/Mul:z:03tf.math.reduce_sum_2/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.reduce_sum_2/SumÓ
IdentityIdentity!tf.math.reduce_sum_2/Sum:output:0^embedding_6/embedding_lookup^embedding_7/embedding_lookup^embedding_8/embedding_lookup*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: :::2<
embedding_6/embedding_lookupembedding_6/embedding_lookup2<
embedding_7/embedding_lookupembedding_7/embedding_lookup2<
embedding_8/embedding_lookupembedding_8/embedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 
×ö
÷
H__inference_custom_model_1_layer_call_and_return_conditional_losses_5920

inputs_0_0

inputs_0_1
inputs_1*
&tf_math_greater_equal_5_greaterequal_y2
.model_2_tf_math_greater_equal_3_greaterequal_y-
)model_2_embedding_8_embedding_lookup_5770-
)model_2_embedding_6_embedding_lookup_5776-
)model_2_embedding_7_embedding_lookup_57842
.model_3_tf_math_greater_equal_4_greaterequal_y.
*model_3_embedding_11_embedding_lookup_5808-
)model_3_embedding_9_embedding_lookup_5814.
*model_3_embedding_10_embedding_lookup_5822.
*tf_clip_by_value_5_clip_by_value_minimum_y&
"tf_clip_by_value_5_clip_by_value_y*
&dense_9_matmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource+
'dense_10_matmul_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource+
'dense_11_matmul_readvariableop_resource,
(dense_11_biasadd_readvariableop_resource+
'dense_13_matmul_readvariableop_resource,
(dense_13_biasadd_readvariableop_resource+
'dense_14_matmul_readvariableop_resource,
(dense_14_biasadd_readvariableop_resource+
'dense_15_matmul_readvariableop_resource,
(dense_15_biasadd_readvariableop_resource+
'dense_16_matmul_readvariableop_resource,
(dense_16_biasadd_readvariableop_resource?
;normalize_1_normalization_1_reshape_readvariableop_resourceA
=normalize_1_normalization_1_reshape_1_readvariableop_resource+
'dense_17_matmul_readvariableop_resource,
(dense_17_biasadd_readvariableop_resource
identity¢dense_10/BiasAdd/ReadVariableOp¢dense_10/MatMul/ReadVariableOp¢dense_11/BiasAdd/ReadVariableOp¢dense_11/MatMul/ReadVariableOp¢dense_12/BiasAdd/ReadVariableOp¢dense_12/MatMul/ReadVariableOp¢dense_13/BiasAdd/ReadVariableOp¢dense_13/MatMul/ReadVariableOp¢dense_14/BiasAdd/ReadVariableOp¢dense_14/MatMul/ReadVariableOp¢dense_15/BiasAdd/ReadVariableOp¢dense_15/MatMul/ReadVariableOp¢dense_16/BiasAdd/ReadVariableOp¢dense_16/MatMul/ReadVariableOp¢dense_17/BiasAdd/ReadVariableOp¢dense_17/MatMul/ReadVariableOp¢dense_9/BiasAdd/ReadVariableOp¢dense_9/MatMul/ReadVariableOp¢$model_2/embedding_6/embedding_lookup¢$model_2/embedding_7/embedding_lookup¢$model_2/embedding_8/embedding_lookup¢%model_3/embedding_10/embedding_lookup¢%model_3/embedding_11/embedding_lookup¢$model_3/embedding_9/embedding_lookup¢2normalize_1/normalization_1/Reshape/ReadVariableOp¢4normalize_1/normalization_1/Reshape_1/ReadVariableOpÀ
$tf.math.greater_equal_5/GreaterEqualGreaterEqualinputs_1&tf_math_greater_equal_5_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2&
$tf.math.greater_equal_5/GreaterEqual
model_2/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
model_2/flatten_2/Const¡
model_2/flatten_2/ReshapeReshape
inputs_0_0 model_2/flatten_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_2/flatten_2/Reshape­
2model_2/tf.clip_by_value_3/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI24
2model_2/tf.clip_by_value_3/clip_by_value/Minimum/y
0model_2/tf.clip_by_value_3/clip_by_value/MinimumMinimum"model_2/flatten_2/Reshape:output:0;model_2/tf.clip_by_value_3/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0model_2/tf.clip_by_value_3/clip_by_value/Minimum
*model_2/tf.clip_by_value_3/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*model_2/tf.clip_by_value_3/clip_by_value/yü
(model_2/tf.clip_by_value_3/clip_by_valueMaximum4model_2/tf.clip_by_value_3/clip_by_value/Minimum:z:03model_2/tf.clip_by_value_3/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(model_2/tf.clip_by_value_3/clip_by_value
+model_2/tf.compat.v1.floor_div_2/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2-
+model_2/tf.compat.v1.floor_div_2/FloorDiv/yø
)model_2/tf.compat.v1.floor_div_2/FloorDivFloorDiv,model_2/tf.clip_by_value_3/clip_by_value:z:04model_2/tf.compat.v1.floor_div_2/FloorDiv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)model_2/tf.compat.v1.floor_div_2/FloorDivò
,model_2/tf.math.greater_equal_3/GreaterEqualGreaterEqual"model_2/flatten_2/Reshape:output:0.model_2_tf_math_greater_equal_3_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,model_2/tf.math.greater_equal_3/GreaterEqual
%model_2/tf.math.floormod_2/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2'
%model_2/tf.math.floormod_2/FloorMod/yæ
#model_2/tf.math.floormod_2/FloorModFloorMod,model_2/tf.clip_by_value_3/clip_by_value:z:0.model_2/tf.math.floormod_2/FloorMod/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#model_2/tf.math.floormod_2/FloorMod«
model_2/embedding_8/CastCast,model_2/tf.clip_by_value_3/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_2/embedding_8/CastÞ
$model_2/embedding_8/embedding_lookupResourceGather)model_2_embedding_8_embedding_lookup_5770model_2/embedding_8/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*<
_class2
0.loc:@model_2/embedding_8/embedding_lookup/5770*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$model_2/embedding_8/embedding_lookup¼
-model_2/embedding_8/embedding_lookup/IdentityIdentity-model_2/embedding_8/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*<
_class2
0.loc:@model_2/embedding_8/embedding_lookup/5770*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-model_2/embedding_8/embedding_lookup/IdentityÝ
/model_2/embedding_8/embedding_lookup/Identity_1Identity6model_2/embedding_8/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/model_2/embedding_8/embedding_lookup/Identity_1¬
model_2/embedding_6/CastCast-model_2/tf.compat.v1.floor_div_2/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_2/embedding_6/CastÞ
$model_2/embedding_6/embedding_lookupResourceGather)model_2_embedding_6_embedding_lookup_5776model_2/embedding_6/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*<
_class2
0.loc:@model_2/embedding_6/embedding_lookup/5776*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$model_2/embedding_6/embedding_lookup¼
-model_2/embedding_6/embedding_lookup/IdentityIdentity-model_2/embedding_6/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*<
_class2
0.loc:@model_2/embedding_6/embedding_lookup/5776*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-model_2/embedding_6/embedding_lookup/IdentityÝ
/model_2/embedding_6/embedding_lookup/Identity_1Identity6model_2/embedding_6/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/model_2/embedding_6/embedding_lookup/Identity_1«
model_2/tf.cast_3/CastCast0model_2/tf.math.greater_equal_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_2/tf.cast_3/Cast
$model_2/tf.__operators__.add_6/AddV2AddV28model_2/embedding_8/embedding_lookup/Identity_1:output:08model_2/embedding_6/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$model_2/tf.__operators__.add_6/AddV2¦
model_2/embedding_7/CastCast'model_2/tf.math.floormod_2/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_2/embedding_7/CastÞ
$model_2/embedding_7/embedding_lookupResourceGather)model_2_embedding_7_embedding_lookup_5784model_2/embedding_7/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*<
_class2
0.loc:@model_2/embedding_7/embedding_lookup/5784*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$model_2/embedding_7/embedding_lookup¼
-model_2/embedding_7/embedding_lookup/IdentityIdentity-model_2/embedding_7/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*<
_class2
0.loc:@model_2/embedding_7/embedding_lookup/5784*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-model_2/embedding_7/embedding_lookup/IdentityÝ
/model_2/embedding_7/embedding_lookup/Identity_1Identity6model_2/embedding_7/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/model_2/embedding_7/embedding_lookup/Identity_1ð
$model_2/tf.__operators__.add_7/AddV2AddV2(model_2/tf.__operators__.add_6/AddV2:z:08model_2/embedding_7/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$model_2/tf.__operators__.add_7/AddV2
'model_2/tf.expand_dims_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2)
'model_2/tf.expand_dims_2/ExpandDims/dimÜ
#model_2/tf.expand_dims_2/ExpandDims
ExpandDimsmodel_2/tf.cast_3/Cast:y:00model_2/tf.expand_dims_2/ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#model_2/tf.expand_dims_2/ExpandDimsÖ
model_2/tf.math.multiply_2/MulMul(model_2/tf.__operators__.add_7/AddV2:z:0,model_2/tf.expand_dims_2/ExpandDims:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
model_2/tf.math.multiply_2/Mulª
2model_2/tf.math.reduce_sum_2/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :24
2model_2/tf.math.reduce_sum_2/Sum/reduction_indicesß
 model_2/tf.math.reduce_sum_2/SumSum"model_2/tf.math.multiply_2/Mul:z:0;model_2/tf.math.reduce_sum_2/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 model_2/tf.math.reduce_sum_2/Sum
model_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
model_3/flatten_3/Const¡
model_3/flatten_3/ReshapeReshape
inputs_0_1 model_3/flatten_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_3/flatten_3/Reshape­
2model_3/tf.clip_by_value_4/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI24
2model_3/tf.clip_by_value_4/clip_by_value/Minimum/y
0model_3/tf.clip_by_value_4/clip_by_value/MinimumMinimum"model_3/flatten_3/Reshape:output:0;model_3/tf.clip_by_value_4/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0model_3/tf.clip_by_value_4/clip_by_value/Minimum
*model_3/tf.clip_by_value_4/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*model_3/tf.clip_by_value_4/clip_by_value/yü
(model_3/tf.clip_by_value_4/clip_by_valueMaximum4model_3/tf.clip_by_value_4/clip_by_value/Minimum:z:03model_3/tf.clip_by_value_4/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(model_3/tf.clip_by_value_4/clip_by_value
+model_3/tf.compat.v1.floor_div_3/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2-
+model_3/tf.compat.v1.floor_div_3/FloorDiv/yø
)model_3/tf.compat.v1.floor_div_3/FloorDivFloorDiv,model_3/tf.clip_by_value_4/clip_by_value:z:04model_3/tf.compat.v1.floor_div_3/FloorDiv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)model_3/tf.compat.v1.floor_div_3/FloorDivò
,model_3/tf.math.greater_equal_4/GreaterEqualGreaterEqual"model_3/flatten_3/Reshape:output:0.model_3_tf_math_greater_equal_4_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,model_3/tf.math.greater_equal_4/GreaterEqual
%model_3/tf.math.floormod_3/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2'
%model_3/tf.math.floormod_3/FloorMod/yæ
#model_3/tf.math.floormod_3/FloorModFloorMod,model_3/tf.clip_by_value_4/clip_by_value:z:0.model_3/tf.math.floormod_3/FloorMod/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#model_3/tf.math.floormod_3/FloorMod­
model_3/embedding_11/CastCast,model_3/tf.clip_by_value_4/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_3/embedding_11/Castã
%model_3/embedding_11/embedding_lookupResourceGather*model_3_embedding_11_embedding_lookup_5808model_3/embedding_11/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*=
_class3
1/loc:@model_3/embedding_11/embedding_lookup/5808*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02'
%model_3/embedding_11/embedding_lookupÀ
.model_3/embedding_11/embedding_lookup/IdentityIdentity.model_3/embedding_11/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*=
_class3
1/loc:@model_3/embedding_11/embedding_lookup/5808*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.model_3/embedding_11/embedding_lookup/Identityà
0model_3/embedding_11/embedding_lookup/Identity_1Identity7model_3/embedding_11/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0model_3/embedding_11/embedding_lookup/Identity_1¬
model_3/embedding_9/CastCast-model_3/tf.compat.v1.floor_div_3/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_3/embedding_9/CastÞ
$model_3/embedding_9/embedding_lookupResourceGather)model_3_embedding_9_embedding_lookup_5814model_3/embedding_9/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*<
_class2
0.loc:@model_3/embedding_9/embedding_lookup/5814*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$model_3/embedding_9/embedding_lookup¼
-model_3/embedding_9/embedding_lookup/IdentityIdentity-model_3/embedding_9/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*<
_class2
0.loc:@model_3/embedding_9/embedding_lookup/5814*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-model_3/embedding_9/embedding_lookup/IdentityÝ
/model_3/embedding_9/embedding_lookup/Identity_1Identity6model_3/embedding_9/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/model_3/embedding_9/embedding_lookup/Identity_1«
model_3/tf.cast_4/CastCast0model_3/tf.math.greater_equal_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_3/tf.cast_4/Cast
$model_3/tf.__operators__.add_8/AddV2AddV29model_3/embedding_11/embedding_lookup/Identity_1:output:08model_3/embedding_9/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$model_3/tf.__operators__.add_8/AddV2¨
model_3/embedding_10/CastCast'model_3/tf.math.floormod_3/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_3/embedding_10/Castã
%model_3/embedding_10/embedding_lookupResourceGather*model_3_embedding_10_embedding_lookup_5822model_3/embedding_10/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*=
_class3
1/loc:@model_3/embedding_10/embedding_lookup/5822*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02'
%model_3/embedding_10/embedding_lookupÀ
.model_3/embedding_10/embedding_lookup/IdentityIdentity.model_3/embedding_10/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*=
_class3
1/loc:@model_3/embedding_10/embedding_lookup/5822*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.model_3/embedding_10/embedding_lookup/Identityà
0model_3/embedding_10/embedding_lookup/Identity_1Identity7model_3/embedding_10/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0model_3/embedding_10/embedding_lookup/Identity_1ñ
$model_3/tf.__operators__.add_9/AddV2AddV2(model_3/tf.__operators__.add_8/AddV2:z:09model_3/embedding_10/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$model_3/tf.__operators__.add_9/AddV2
'model_3/tf.expand_dims_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2)
'model_3/tf.expand_dims_3/ExpandDims/dimÜ
#model_3/tf.expand_dims_3/ExpandDims
ExpandDimsmodel_3/tf.cast_4/Cast:y:00model_3/tf.expand_dims_3/ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#model_3/tf.expand_dims_3/ExpandDimsÖ
model_3/tf.math.multiply_3/MulMul(model_3/tf.__operators__.add_9/AddV2:z:0,model_3/tf.expand_dims_3/ExpandDims:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
model_3/tf.math.multiply_3/Mulª
2model_3/tf.math.reduce_sum_3/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :24
2model_3/tf.math.reduce_sum_3/Sum/reduction_indicesß
 model_3/tf.math.reduce_sum_3/SumSum"model_3/tf.math.multiply_3/Mul:z:0;model_3/tf.math.reduce_sum_3/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 model_3/tf.math.reduce_sum_3/SumÇ
(tf.clip_by_value_5/clip_by_value/MinimumMinimuminputs_1*tf_clip_by_value_5_clip_by_value_minimum_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2*
(tf.clip_by_value_5/clip_by_value/MinimumÓ
 tf.clip_by_value_5/clip_by_valueMaximum,tf.clip_by_value_5/clip_by_value/Minimum:z:0"tf_clip_by_value_5_clip_by_value_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2"
 tf.clip_by_value_5/clip_by_value
tf.cast_5/CastCast(tf.math.greater_equal_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
tf.cast_5/Castt
tf.concat_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_3/concat/axisè
tf.concat_3/concatConcatV2)model_2/tf.math.reduce_sum_2/Sum:output:0)model_3/tf.math.reduce_sum_3/Sum:output:0 tf.concat_3/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.concat_3/concat}
tf.concat_4/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
tf.concat_4/concat/axisË
tf.concat_4/concatConcatV2$tf.clip_by_value_5/clip_by_value:z:0tf.cast_5/Cast:y:0 tf.concat_4/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.concat_4/concat§
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_9/MatMul/ReadVariableOp¡
dense_9/MatMulMatMultf.concat_3/concat:output:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_9/MatMul¥
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_9/BiasAdd/ReadVariableOp¢
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_9/BiasAddq
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_9/Relu©
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02 
dense_12/MatMul/ReadVariableOp¤
dense_12/MatMulMatMultf.concat_4/concat:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_12/MatMul¨
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_12/BiasAdd/ReadVariableOp¦
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_12/BiasAddª
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_10/MatMul/ReadVariableOp£
dense_10/MatMulMatMuldense_9/Relu:activations:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_10/MatMul¨
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_10/BiasAdd/ReadVariableOp¦
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_10/BiasAddt
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_10/Reluª
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_11/MatMul/ReadVariableOp¤
dense_11/MatMulMatMuldense_10/Relu:activations:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_11/MatMul¨
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_11/BiasAdd/ReadVariableOp¦
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_11/BiasAddt
dense_11/ReluReludense_11/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_11/Reluª
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_13/MatMul/ReadVariableOp¢
dense_13/MatMulMatMuldense_12/BiasAdd:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_13/MatMul¨
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_13/BiasAdd/ReadVariableOp¦
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_13/BiasAdd}
tf.concat_5/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
tf.concat_5/concat/axisÊ
tf.concat_5/concatConcatV2dense_11/Relu:activations:0dense_13/BiasAdd:output:0 tf.concat_5/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.concat_5/concatª
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_14/MatMul/ReadVariableOp¤
dense_14/MatMulMatMultf.concat_5/concat:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_14/MatMul¨
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_14/BiasAdd/ReadVariableOp¦
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_14/BiasAdd|
tf.nn.relu_3/ReluReludense_14/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.nn.relu_3/Reluª
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_15/MatMul/ReadVariableOp¨
dense_15/MatMulMatMultf.nn.relu_3/Relu:activations:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_15/MatMul¨
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_15/BiasAdd/ReadVariableOp¦
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_15/BiasAdd¶
tf.__operators__.add_10/AddV2AddV2dense_15/BiasAdd:output:0tf.nn.relu_3/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_10/AddV2
tf.nn.relu_4/ReluRelu!tf.__operators__.add_10/AddV2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.nn.relu_4/Reluª
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_16/MatMul/ReadVariableOp¨
dense_16/MatMulMatMultf.nn.relu_4/Relu:activations:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_16/MatMul¨
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_16/BiasAdd/ReadVariableOp¦
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_16/BiasAdd¶
tf.__operators__.add_11/AddV2AddV2dense_16/BiasAdd:output:0tf.nn.relu_4/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_11/AddV2
tf.nn.relu_5/ReluRelu!tf.__operators__.add_11/AddV2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.nn.relu_5/Reluá
2normalize_1/normalization_1/Reshape/ReadVariableOpReadVariableOp;normalize_1_normalization_1_reshape_readvariableop_resource*
_output_shapes	
:*
dtype024
2normalize_1/normalization_1/Reshape/ReadVariableOp§
)normalize_1/normalization_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2+
)normalize_1/normalization_1/Reshape/shapeï
#normalize_1/normalization_1/ReshapeReshape:normalize_1/normalization_1/Reshape/ReadVariableOp:value:02normalize_1/normalization_1/Reshape/shape:output:0*
T0*
_output_shapes
:	2%
#normalize_1/normalization_1/Reshapeç
4normalize_1/normalization_1/Reshape_1/ReadVariableOpReadVariableOp=normalize_1_normalization_1_reshape_1_readvariableop_resource*
_output_shapes	
:*
dtype026
4normalize_1/normalization_1/Reshape_1/ReadVariableOp«
+normalize_1/normalization_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2-
+normalize_1/normalization_1/Reshape_1/shape÷
%normalize_1/normalization_1/Reshape_1Reshape<normalize_1/normalization_1/Reshape_1/ReadVariableOp:value:04normalize_1/normalization_1/Reshape_1/shape:output:0*
T0*
_output_shapes
:	2'
%normalize_1/normalization_1/Reshape_1Ë
normalize_1/normalization_1/subSubtf.nn.relu_5/Relu:activations:0,normalize_1/normalization_1/Reshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
normalize_1/normalization_1/sub¦
 normalize_1/normalization_1/SqrtSqrt.normalize_1/normalization_1/Reshape_1:output:0*
T0*
_output_shapes
:	2"
 normalize_1/normalization_1/Sqrt
%normalize_1/normalization_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö32'
%normalize_1/normalization_1/Maximum/yÕ
#normalize_1/normalization_1/MaximumMaximum$normalize_1/normalization_1/Sqrt:y:0.normalize_1/normalization_1/Maximum/y:output:0*
T0*
_output_shapes
:	2%
#normalize_1/normalization_1/MaximumÖ
#normalize_1/normalization_1/truedivRealDiv#normalize_1/normalization_1/sub:z:0'normalize_1/normalization_1/Maximum:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#normalize_1/normalization_1/truediv©
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02 
dense_17/MatMul/ReadVariableOp¯
dense_17/MatMulMatMul'normalize_1/normalization_1/truediv:z:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_17/MatMul§
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_17/BiasAdd/ReadVariableOp¥
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_17/BiasAdd
IdentityIdentitydense_17/BiasAdd:output:0 ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp%^model_2/embedding_6/embedding_lookup%^model_2/embedding_7/embedding_lookup%^model_2/embedding_8/embedding_lookup&^model_3/embedding_10/embedding_lookup&^model_3/embedding_11/embedding_lookup%^model_3/embedding_9/embedding_lookup3^normalize_1/normalization_1/Reshape/ReadVariableOp5^normalize_1/normalization_1/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*À
_input_shapes®
«:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
: : :::: :::: : ::::::::::::::::::::2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2L
$model_2/embedding_6/embedding_lookup$model_2/embedding_6/embedding_lookup2L
$model_2/embedding_7/embedding_lookup$model_2/embedding_7/embedding_lookup2L
$model_2/embedding_8/embedding_lookup$model_2/embedding_8/embedding_lookup2N
%model_3/embedding_10/embedding_lookup%model_3/embedding_10/embedding_lookup2N
%model_3/embedding_11/embedding_lookup%model_3/embedding_11/embedding_lookup2L
$model_3/embedding_9/embedding_lookup$model_3/embedding_9/embedding_lookup2h
2normalize_1/normalization_1/Reshape/ReadVariableOp2normalize_1/normalization_1/Reshape/ReadVariableOp2l
4normalize_1/normalization_1/Reshape_1/ReadVariableOp4normalize_1/normalization_1/Reshape_1/ReadVariableOp:S O
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/0/0:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/0/1:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
	
Û
B__inference_dense_15_layer_call_and_return_conditional_losses_4988

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô
Ô
"__inference_signature_wrapper_5580
bets

cards0

cards1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29
identity¢StatefulPartitionedCallì
StatefulPartitionedCallStatefulPartitionedCallcards0cards1betsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
 !*2
config_proto" 

CPU

GPU2 *0J 8 *(
f#R!
__inference__wrapped_model_42772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*À
_input_shapes®
«:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : :::: :::: : ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:M I
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


_user_specified_namebets:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namecards0:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namecards1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
	
Û
B__inference_dense_17_layer_call_and_return_conditional_losses_5077

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
õ	
Û
B__inference_dense_10_layer_call_and_return_conditional_losses_6309

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
-
Ä
A__inference_model_3_layer_call_and_return_conditional_losses_4718

inputs*
&tf_math_greater_equal_4_greaterequal_y
embedding_11_4700
embedding_9_4703
embedding_10_4708
identity¢$embedding_10/StatefulPartitionedCall¢$embedding_11/StatefulPartitionedCall¢#embedding_9/StatefulPartitionedCallÚ
flatten_3/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_flatten_3_layer_call_and_return_conditional_losses_45132
flatten_3/PartitionedCall
*tf.clip_by_value_4/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_4/clip_by_value/Minimum/yê
(tf.clip_by_value_4/clip_by_value/MinimumMinimum"flatten_3/PartitionedCall:output:03tf.clip_by_value_4/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(tf.clip_by_value_4/clip_by_value/Minimum
"tf.clip_by_value_4/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_4/clip_by_value/yÜ
 tf.clip_by_value_4/clip_by_valueMaximum,tf.clip_by_value_4/clip_by_value/Minimum:z:0+tf.clip_by_value_4/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 tf.clip_by_value_4/clip_by_value
#tf.compat.v1.floor_div_3/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2%
#tf.compat.v1.floor_div_3/FloorDiv/yØ
!tf.compat.v1.floor_div_3/FloorDivFloorDiv$tf.clip_by_value_4/clip_by_value:z:0,tf.compat.v1.floor_div_3/FloorDiv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!tf.compat.v1.floor_div_3/FloorDivÚ
$tf.math.greater_equal_4/GreaterEqualGreaterEqual"flatten_3/PartitionedCall:output:0&tf_math_greater_equal_4_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$tf.math.greater_equal_4/GreaterEqual
tf.math.floormod_3/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2
tf.math.floormod_3/FloorMod/yÆ
tf.math.floormod_3/FloorModFloorMod$tf.clip_by_value_4/clip_by_value:z:0&tf.math.floormod_3/FloorMod/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.floormod_3/FloorModµ
$embedding_11/StatefulPartitionedCallStatefulPartitionedCall$tf.clip_by_value_4/clip_by_value:z:0embedding_11_4700*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_embedding_11_layer_call_and_return_conditional_losses_45412&
$embedding_11/StatefulPartitionedCall²
#embedding_9/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_3/FloorDiv:z:0embedding_9_4703*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_embedding_9_layer_call_and_return_conditional_losses_45632%
#embedding_9/StatefulPartitionedCall
tf.cast_4/CastCast(tf.math.greater_equal_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.cast_4/CastÙ
tf.__operators__.add_8/AddV2AddV2-embedding_11/StatefulPartitionedCall:output:0,embedding_9/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_8/AddV2°
$embedding_10/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_3/FloorMod:z:0embedding_10_4708*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_embedding_10_layer_call_and_return_conditional_losses_45872&
$embedding_10/StatefulPartitionedCallÍ
tf.__operators__.add_9/AddV2AddV2 tf.__operators__.add_8/AddV2:z:0-embedding_10/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_9/AddV2
tf.expand_dims_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
tf.expand_dims_3/ExpandDims/dim¼
tf.expand_dims_3/ExpandDims
ExpandDimstf.cast_4/Cast:y:0(tf.expand_dims_3/ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.expand_dims_3/ExpandDims¶
tf.math.multiply_3/MulMul tf.__operators__.add_9/AddV2:z:0$tf.expand_dims_3/ExpandDims:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.multiply_3/Mul
*tf.math.reduce_sum_3/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_3/Sum/reduction_indices¿
tf.math.reduce_sum_3/SumSumtf.math.multiply_3/Mul:z:03tf.math.reduce_sum_3/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.reduce_sum_3/Sumê
IdentityIdentity!tf.math.reduce_sum_3/Sum:output:0%^embedding_10/StatefulPartitionedCall%^embedding_11/StatefulPartitionedCall$^embedding_9/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: :::2L
$embedding_10/StatefulPartitionedCall$embedding_10/StatefulPartitionedCall2L
$embedding_11/StatefulPartitionedCall$embedding_11/StatefulPartitionedCall2J
#embedding_9/StatefulPartitionedCall#embedding_9/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 
Ù	

E__inference_embedding_6_layer_call_and_return_conditional_losses_6516

inputs
embedding_lookup_6510
identity¢embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Castú
embedding_lookupResourceGatherembedding_lookup_6510Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*(
_class
loc:@embedding_lookup/6510*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_lookupì
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@embedding_lookup/6510*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity¡
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

D
(__inference_flatten_2_layer_call_fn_6489

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_flatten_2_layer_call_and_return_conditional_losses_42872
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
á
|
'__inference_dense_11_layer_call_fn_6357

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_dense_11_layer_call_and_return_conditional_losses_49072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Û
B__inference_dense_13_layer_call_and_return_conditional_losses_4933

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù	

E__inference_embedding_8_layer_call_and_return_conditional_losses_4315

inputs
embedding_lookup_4309
identity¢embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Castú
embedding_lookupResourceGatherembedding_lookup_4309Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*(
_class
loc:@embedding_lookup/4309*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_lookupì
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@embedding_lookup/4309*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity¡
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
á
|
'__inference_dense_14_layer_call_fn_6395

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_dense_14_layer_call_and_return_conditional_losses_49612
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
©
_
C__inference_flatten_2_layer_call_and_return_conditional_losses_4287

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ú	

F__inference_embedding_10_layer_call_and_return_conditional_losses_4587

inputs
embedding_lookup_4581
identity¢embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Castú
embedding_lookupResourceGatherembedding_lookup_4581Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*(
_class
loc:@embedding_lookup/4581*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_lookupì
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@embedding_lookup/4581*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity¡
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
õ	
Û
B__inference_dense_11_layer_call_and_return_conditional_losses_4907

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬V
µ	
H__inference_custom_model_1_layer_call_and_return_conditional_losses_5186

cards0

cards1
bets*
&tf_math_greater_equal_5_greaterequal_y
model_2_5101
model_2_5103
model_2_5105
model_2_5107
model_3_5110
model_3_5112
model_3_5114
model_3_5116.
*tf_clip_by_value_5_clip_by_value_minimum_y&
"tf_clip_by_value_5_clip_by_value_y
dense_9_5128
dense_9_5130
dense_12_5133
dense_12_5135
dense_10_5138
dense_10_5140
dense_11_5143
dense_11_5145
dense_13_5148
dense_13_5150
dense_14_5155
dense_14_5157
dense_15_5161
dense_15_5163
dense_16_5168
dense_16_5170
normalize_1_5175
normalize_1_5177
dense_17_5180
dense_17_5182
identity¢ dense_10/StatefulPartitionedCall¢ dense_11/StatefulPartitionedCall¢ dense_12/StatefulPartitionedCall¢ dense_13/StatefulPartitionedCall¢ dense_14/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall¢ dense_16/StatefulPartitionedCall¢ dense_17/StatefulPartitionedCall¢dense_9/StatefulPartitionedCall¢model_2/StatefulPartitionedCall¢model_3/StatefulPartitionedCall¢#normalize_1/StatefulPartitionedCall¼
$tf.math.greater_equal_5/GreaterEqualGreaterEqualbets&tf_math_greater_equal_5_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2&
$tf.math.greater_equal_5/GreaterEqual®
model_2/StatefulPartitionedCallStatefulPartitionedCallcards0model_2_5101model_2_5103model_2_5105model_2_5107*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_model_2_layer_call_and_return_conditional_losses_44922!
model_2/StatefulPartitionedCall®
model_3/StatefulPartitionedCallStatefulPartitionedCallcards1model_3_5110model_3_5112model_3_5114model_3_5116*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_model_3_layer_call_and_return_conditional_losses_47182!
model_3/StatefulPartitionedCallÃ
(tf.clip_by_value_5/clip_by_value/MinimumMinimumbets*tf_clip_by_value_5_clip_by_value_minimum_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2*
(tf.clip_by_value_5/clip_by_value/MinimumÓ
 tf.clip_by_value_5/clip_by_valueMaximum,tf.clip_by_value_5/clip_by_value/Minimum:z:0"tf_clip_by_value_5_clip_by_value_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2"
 tf.clip_by_value_5/clip_by_value
tf.cast_5/CastCast(tf.math.greater_equal_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
tf.cast_5/Castt
tf.concat_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_3/concat/axisæ
tf.concat_3/concatConcatV2(model_2/StatefulPartitionedCall:output:0(model_3/StatefulPartitionedCall:output:0 tf.concat_3/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.concat_3/concat}
tf.concat_4/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
tf.concat_4/concat/axisË
tf.concat_4/concatConcatV2$tf.clip_by_value_5/clip_by_value:z:0tf.cast_5/Cast:y:0 tf.concat_4/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.concat_4/concat¤
dense_9/StatefulPartitionedCallStatefulPartitionedCalltf.concat_3/concat:output:0dense_9_5128dense_9_5130*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_dense_9_layer_call_and_return_conditional_losses_48272!
dense_9/StatefulPartitionedCall©
 dense_12/StatefulPartitionedCallStatefulPartitionedCalltf.concat_4/concat:output:0dense_12_5133dense_12_5135*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_dense_12_layer_call_and_return_conditional_losses_48532"
 dense_12/StatefulPartitionedCall¶
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0dense_10_5138dense_10_5140*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_dense_10_layer_call_and_return_conditional_losses_48802"
 dense_10/StatefulPartitionedCall·
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_5143dense_11_5145*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_dense_11_layer_call_and_return_conditional_losses_49072"
 dense_11/StatefulPartitionedCall·
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_5148dense_13_5150*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_dense_13_layer_call_and_return_conditional_losses_49332"
 dense_13/StatefulPartitionedCall}
tf.concat_5/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
tf.concat_5/concat/axisè
tf.concat_5/concatConcatV2)dense_11/StatefulPartitionedCall:output:0)dense_13/StatefulPartitionedCall:output:0 tf.concat_5/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.concat_5/concat©
 dense_14/StatefulPartitionedCallStatefulPartitionedCalltf.concat_5/concat:output:0dense_14_5155dense_14_5157*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_dense_14_layer_call_and_return_conditional_losses_49612"
 dense_14/StatefulPartitionedCall
tf.nn.relu_3/ReluRelu)dense_14/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.nn.relu_3/Relu­
 dense_15/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_3/Relu:activations:0dense_15_5161dense_15_5163*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_dense_15_layer_call_and_return_conditional_losses_49882"
 dense_15/StatefulPartitionedCallÆ
tf.__operators__.add_10/AddV2AddV2)dense_15/StatefulPartitionedCall:output:0tf.nn.relu_3/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_10/AddV2
tf.nn.relu_4/ReluRelu!tf.__operators__.add_10/AddV2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.nn.relu_4/Relu­
 dense_16/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_4/Relu:activations:0dense_16_5168dense_16_5170*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_dense_16_layer_call_and_return_conditional_losses_50162"
 dense_16/StatefulPartitionedCallÆ
tf.__operators__.add_11/AddV2AddV2)dense_16/StatefulPartitionedCall:output:0tf.nn.relu_4/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_11/AddV2
tf.nn.relu_5/ReluRelu!tf.__operators__.add_11/AddV2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.nn.relu_5/Relu¼
#normalize_1/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_5/Relu:activations:0normalize_1_5175normalize_1_5177*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_normalize_1_layer_call_and_return_conditional_losses_50512%
#normalize_1/StatefulPartitionedCall¹
 dense_17/StatefulPartitionedCallStatefulPartitionedCall,normalize_1/StatefulPartitionedCall:output:0dense_17_5180dense_17_5182*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_dense_17_layer_call_and_return_conditional_losses_50772"
 dense_17/StatefulPartitionedCall¡
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall ^model_2/StatefulPartitionedCall ^model_3/StatefulPartitionedCall$^normalize_1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*À
_input_shapes®
«:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
: : :::: :::: : ::::::::::::::::::::2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2B
model_2/StatefulPartitionedCallmodel_2/StatefulPartitionedCall2B
model_3/StatefulPartitionedCallmodel_3/StatefulPartitionedCall2J
#normalize_1/StatefulPartitionedCall#normalize_1/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namecards0:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namecards1:MI
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


_user_specified_namebets:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
8
Þ
A__inference_model_2_layer_call_and_return_conditional_losses_6100

inputs*
&tf_math_greater_equal_3_greaterequal_y%
!embedding_8_embedding_lookup_6074%
!embedding_6_embedding_lookup_6080%
!embedding_7_embedding_lookup_6088
identity¢embedding_6/embedding_lookup¢embedding_7/embedding_lookup¢embedding_8/embedding_lookups
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten_2/Const
flatten_2/ReshapeReshapeinputsflatten_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten_2/Reshape
*tf.clip_by_value_3/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_3/clip_by_value/Minimum/yâ
(tf.clip_by_value_3/clip_by_value/MinimumMinimumflatten_2/Reshape:output:03tf.clip_by_value_3/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(tf.clip_by_value_3/clip_by_value/Minimum
"tf.clip_by_value_3/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_3/clip_by_value/yÜ
 tf.clip_by_value_3/clip_by_valueMaximum,tf.clip_by_value_3/clip_by_value/Minimum:z:0+tf.clip_by_value_3/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 tf.clip_by_value_3/clip_by_value
#tf.compat.v1.floor_div_2/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2%
#tf.compat.v1.floor_div_2/FloorDiv/yØ
!tf.compat.v1.floor_div_2/FloorDivFloorDiv$tf.clip_by_value_3/clip_by_value:z:0,tf.compat.v1.floor_div_2/FloorDiv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!tf.compat.v1.floor_div_2/FloorDivÒ
$tf.math.greater_equal_3/GreaterEqualGreaterEqualflatten_2/Reshape:output:0&tf_math_greater_equal_3_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$tf.math.greater_equal_3/GreaterEqual
tf.math.floormod_2/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2
tf.math.floormod_2/FloorMod/yÆ
tf.math.floormod_2/FloorModFloorMod$tf.clip_by_value_3/clip_by_value:z:0&tf.math.floormod_2/FloorMod/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.floormod_2/FloorMod
embedding_8/CastCast$tf.clip_by_value_3/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_8/Cast¶
embedding_8/embedding_lookupResourceGather!embedding_8_embedding_lookup_6074embedding_8/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*4
_class*
(&loc:@embedding_8/embedding_lookup/6074*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_8/embedding_lookup
%embedding_8/embedding_lookup/IdentityIdentity%embedding_8/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*4
_class*
(&loc:@embedding_8/embedding_lookup/6074*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%embedding_8/embedding_lookup/IdentityÅ
'embedding_8/embedding_lookup/Identity_1Identity.embedding_8/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'embedding_8/embedding_lookup/Identity_1
embedding_6/CastCast%tf.compat.v1.floor_div_2/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_6/Cast¶
embedding_6/embedding_lookupResourceGather!embedding_6_embedding_lookup_6080embedding_6/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*4
_class*
(&loc:@embedding_6/embedding_lookup/6080*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_6/embedding_lookup
%embedding_6/embedding_lookup/IdentityIdentity%embedding_6/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*4
_class*
(&loc:@embedding_6/embedding_lookup/6080*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%embedding_6/embedding_lookup/IdentityÅ
'embedding_6/embedding_lookup/Identity_1Identity.embedding_6/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'embedding_6/embedding_lookup/Identity_1
tf.cast_3/CastCast(tf.math.greater_equal_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.cast_3/Castà
tf.__operators__.add_6/AddV2AddV20embedding_8/embedding_lookup/Identity_1:output:00embedding_6/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_6/AddV2
embedding_7/CastCasttf.math.floormod_2/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_7/Cast¶
embedding_7/embedding_lookupResourceGather!embedding_7_embedding_lookup_6088embedding_7/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*4
_class*
(&loc:@embedding_7/embedding_lookup/6088*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_7/embedding_lookup
%embedding_7/embedding_lookup/IdentityIdentity%embedding_7/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*4
_class*
(&loc:@embedding_7/embedding_lookup/6088*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%embedding_7/embedding_lookup/IdentityÅ
'embedding_7/embedding_lookup/Identity_1Identity.embedding_7/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'embedding_7/embedding_lookup/Identity_1Ð
tf.__operators__.add_7/AddV2AddV2 tf.__operators__.add_6/AddV2:z:00embedding_7/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_7/AddV2
tf.expand_dims_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
tf.expand_dims_2/ExpandDims/dim¼
tf.expand_dims_2/ExpandDims
ExpandDimstf.cast_3/Cast:y:0(tf.expand_dims_2/ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.expand_dims_2/ExpandDims¶
tf.math.multiply_2/MulMul tf.__operators__.add_7/AddV2:z:0$tf.expand_dims_2/ExpandDims:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.multiply_2/Mul
*tf.math.reduce_sum_2/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_2/Sum/reduction_indices¿
tf.math.reduce_sum_2/SumSumtf.math.multiply_2/Mul:z:03tf.math.reduce_sum_2/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.reduce_sum_2/SumÓ
IdentityIdentity!tf.math.reduce_sum_2/Sum:output:0^embedding_6/embedding_lookup^embedding_7/embedding_lookup^embedding_8/embedding_lookup*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: :::2<
embedding_6/embedding_lookupembedding_6/embedding_lookup2<
embedding_7/embedding_lookupembedding_7/embedding_lookup2<
embedding_8/embedding_lookupembedding_8/embedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 
	
Û
B__inference_dense_14_layer_call_and_return_conditional_losses_4961

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°8
â
A__inference_model_3_layer_call_and_return_conditional_losses_6210

inputs*
&tf_math_greater_equal_4_greaterequal_y&
"embedding_11_embedding_lookup_6184%
!embedding_9_embedding_lookup_6190&
"embedding_10_embedding_lookup_6198
identity¢embedding_10/embedding_lookup¢embedding_11/embedding_lookup¢embedding_9/embedding_lookups
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten_3/Const
flatten_3/ReshapeReshapeinputsflatten_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten_3/Reshape
*tf.clip_by_value_4/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_4/clip_by_value/Minimum/yâ
(tf.clip_by_value_4/clip_by_value/MinimumMinimumflatten_3/Reshape:output:03tf.clip_by_value_4/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(tf.clip_by_value_4/clip_by_value/Minimum
"tf.clip_by_value_4/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_4/clip_by_value/yÜ
 tf.clip_by_value_4/clip_by_valueMaximum,tf.clip_by_value_4/clip_by_value/Minimum:z:0+tf.clip_by_value_4/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 tf.clip_by_value_4/clip_by_value
#tf.compat.v1.floor_div_3/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2%
#tf.compat.v1.floor_div_3/FloorDiv/yØ
!tf.compat.v1.floor_div_3/FloorDivFloorDiv$tf.clip_by_value_4/clip_by_value:z:0,tf.compat.v1.floor_div_3/FloorDiv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!tf.compat.v1.floor_div_3/FloorDivÒ
$tf.math.greater_equal_4/GreaterEqualGreaterEqualflatten_3/Reshape:output:0&tf_math_greater_equal_4_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$tf.math.greater_equal_4/GreaterEqual
tf.math.floormod_3/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2
tf.math.floormod_3/FloorMod/yÆ
tf.math.floormod_3/FloorModFloorMod$tf.clip_by_value_4/clip_by_value:z:0&tf.math.floormod_3/FloorMod/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.floormod_3/FloorMod
embedding_11/CastCast$tf.clip_by_value_4/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_11/Cast»
embedding_11/embedding_lookupResourceGather"embedding_11_embedding_lookup_6184embedding_11/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*5
_class+
)'loc:@embedding_11/embedding_lookup/6184*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_11/embedding_lookup 
&embedding_11/embedding_lookup/IdentityIdentity&embedding_11/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*5
_class+
)'loc:@embedding_11/embedding_lookup/6184*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&embedding_11/embedding_lookup/IdentityÈ
(embedding_11/embedding_lookup/Identity_1Identity/embedding_11/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(embedding_11/embedding_lookup/Identity_1
embedding_9/CastCast%tf.compat.v1.floor_div_3/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_9/Cast¶
embedding_9/embedding_lookupResourceGather!embedding_9_embedding_lookup_6190embedding_9/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*4
_class*
(&loc:@embedding_9/embedding_lookup/6190*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_9/embedding_lookup
%embedding_9/embedding_lookup/IdentityIdentity%embedding_9/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*4
_class*
(&loc:@embedding_9/embedding_lookup/6190*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%embedding_9/embedding_lookup/IdentityÅ
'embedding_9/embedding_lookup/Identity_1Identity.embedding_9/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'embedding_9/embedding_lookup/Identity_1
tf.cast_4/CastCast(tf.math.greater_equal_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.cast_4/Castá
tf.__operators__.add_8/AddV2AddV21embedding_11/embedding_lookup/Identity_1:output:00embedding_9/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_8/AddV2
embedding_10/CastCasttf.math.floormod_3/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_10/Cast»
embedding_10/embedding_lookupResourceGather"embedding_10_embedding_lookup_6198embedding_10/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*5
_class+
)'loc:@embedding_10/embedding_lookup/6198*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_10/embedding_lookup 
&embedding_10/embedding_lookup/IdentityIdentity&embedding_10/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*5
_class+
)'loc:@embedding_10/embedding_lookup/6198*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&embedding_10/embedding_lookup/IdentityÈ
(embedding_10/embedding_lookup/Identity_1Identity/embedding_10/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(embedding_10/embedding_lookup/Identity_1Ñ
tf.__operators__.add_9/AddV2AddV2 tf.__operators__.add_8/AddV2:z:01embedding_10/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_9/AddV2
tf.expand_dims_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
tf.expand_dims_3/ExpandDims/dim¼
tf.expand_dims_3/ExpandDims
ExpandDimstf.cast_4/Cast:y:0(tf.expand_dims_3/ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.expand_dims_3/ExpandDims¶
tf.math.multiply_3/MulMul tf.__operators__.add_9/AddV2:z:0$tf.expand_dims_3/ExpandDims:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.multiply_3/Mul
*tf.math.reduce_sum_3/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_3/Sum/reduction_indices¿
tf.math.reduce_sum_3/SumSumtf.math.multiply_3/Mul:z:03tf.math.reduce_sum_3/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.reduce_sum_3/SumÕ
IdentityIdentity!tf.math.reduce_sum_3/Sum:output:0^embedding_10/embedding_lookup^embedding_11/embedding_lookup^embedding_9/embedding_lookup*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: :::2>
embedding_10/embedding_lookupembedding_10/embedding_lookup2>
embedding_11/embedding_lookupembedding_11/embedding_lookup2<
embedding_9/embedding_lookupembedding_9/embedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 
-
Å
A__inference_model_3_layer_call_and_return_conditional_losses_4638
input_4*
&tf_math_greater_equal_4_greaterequal_y
embedding_11_4620
embedding_9_4623
embedding_10_4628
identity¢$embedding_10/StatefulPartitionedCall¢$embedding_11/StatefulPartitionedCall¢#embedding_9/StatefulPartitionedCallÛ
flatten_3/PartitionedCallPartitionedCallinput_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_flatten_3_layer_call_and_return_conditional_losses_45132
flatten_3/PartitionedCall
*tf.clip_by_value_4/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_4/clip_by_value/Minimum/yê
(tf.clip_by_value_4/clip_by_value/MinimumMinimum"flatten_3/PartitionedCall:output:03tf.clip_by_value_4/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(tf.clip_by_value_4/clip_by_value/Minimum
"tf.clip_by_value_4/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_4/clip_by_value/yÜ
 tf.clip_by_value_4/clip_by_valueMaximum,tf.clip_by_value_4/clip_by_value/Minimum:z:0+tf.clip_by_value_4/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 tf.clip_by_value_4/clip_by_value
#tf.compat.v1.floor_div_3/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2%
#tf.compat.v1.floor_div_3/FloorDiv/yØ
!tf.compat.v1.floor_div_3/FloorDivFloorDiv$tf.clip_by_value_4/clip_by_value:z:0,tf.compat.v1.floor_div_3/FloorDiv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!tf.compat.v1.floor_div_3/FloorDivÚ
$tf.math.greater_equal_4/GreaterEqualGreaterEqual"flatten_3/PartitionedCall:output:0&tf_math_greater_equal_4_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$tf.math.greater_equal_4/GreaterEqual
tf.math.floormod_3/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2
tf.math.floormod_3/FloorMod/yÆ
tf.math.floormod_3/FloorModFloorMod$tf.clip_by_value_4/clip_by_value:z:0&tf.math.floormod_3/FloorMod/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.floormod_3/FloorModµ
$embedding_11/StatefulPartitionedCallStatefulPartitionedCall$tf.clip_by_value_4/clip_by_value:z:0embedding_11_4620*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_embedding_11_layer_call_and_return_conditional_losses_45412&
$embedding_11/StatefulPartitionedCall²
#embedding_9/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_3/FloorDiv:z:0embedding_9_4623*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_embedding_9_layer_call_and_return_conditional_losses_45632%
#embedding_9/StatefulPartitionedCall
tf.cast_4/CastCast(tf.math.greater_equal_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.cast_4/CastÙ
tf.__operators__.add_8/AddV2AddV2-embedding_11/StatefulPartitionedCall:output:0,embedding_9/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_8/AddV2°
$embedding_10/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_3/FloorMod:z:0embedding_10_4628*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_embedding_10_layer_call_and_return_conditional_losses_45872&
$embedding_10/StatefulPartitionedCallÍ
tf.__operators__.add_9/AddV2AddV2 tf.__operators__.add_8/AddV2:z:0-embedding_10/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_9/AddV2
tf.expand_dims_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
tf.expand_dims_3/ExpandDims/dim¼
tf.expand_dims_3/ExpandDims
ExpandDimstf.cast_4/Cast:y:0(tf.expand_dims_3/ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.expand_dims_3/ExpandDims¶
tf.math.multiply_3/MulMul tf.__operators__.add_9/AddV2:z:0$tf.expand_dims_3/ExpandDims:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.multiply_3/Mul
*tf.math.reduce_sum_3/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_3/Sum/reduction_indices¿
tf.math.reduce_sum_3/SumSumtf.math.multiply_3/Mul:z:03tf.math.reduce_sum_3/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.reduce_sum_3/Sumê
IdentityIdentity!tf.math.reduce_sum_3/Sum:output:0%^embedding_10/StatefulPartitionedCall%^embedding_11/StatefulPartitionedCall$^embedding_9/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: :::2L
$embedding_10/StatefulPartitionedCall$embedding_10/StatefulPartitionedCall2L
$embedding_11/StatefulPartitionedCall$embedding_11/StatefulPartitionedCall2J
#embedding_9/StatefulPartitionedCall#embedding_9/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4:

_output_shapes
: 
¼

&__inference_model_3_layer_call_fn_4684
input_4
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_model_3_layer_call_and_return_conditional_losses_46732
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: :::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4:

_output_shapes
: 
-
Å
A__inference_model_3_layer_call_and_return_conditional_losses_4606
input_4*
&tf_math_greater_equal_4_greaterequal_y
embedding_11_4550
embedding_9_4572
embedding_10_4596
identity¢$embedding_10/StatefulPartitionedCall¢$embedding_11/StatefulPartitionedCall¢#embedding_9/StatefulPartitionedCallÛ
flatten_3/PartitionedCallPartitionedCallinput_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_flatten_3_layer_call_and_return_conditional_losses_45132
flatten_3/PartitionedCall
*tf.clip_by_value_4/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_4/clip_by_value/Minimum/yê
(tf.clip_by_value_4/clip_by_value/MinimumMinimum"flatten_3/PartitionedCall:output:03tf.clip_by_value_4/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(tf.clip_by_value_4/clip_by_value/Minimum
"tf.clip_by_value_4/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_4/clip_by_value/yÜ
 tf.clip_by_value_4/clip_by_valueMaximum,tf.clip_by_value_4/clip_by_value/Minimum:z:0+tf.clip_by_value_4/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 tf.clip_by_value_4/clip_by_value
#tf.compat.v1.floor_div_3/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2%
#tf.compat.v1.floor_div_3/FloorDiv/yØ
!tf.compat.v1.floor_div_3/FloorDivFloorDiv$tf.clip_by_value_4/clip_by_value:z:0,tf.compat.v1.floor_div_3/FloorDiv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!tf.compat.v1.floor_div_3/FloorDivÚ
$tf.math.greater_equal_4/GreaterEqualGreaterEqual"flatten_3/PartitionedCall:output:0&tf_math_greater_equal_4_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$tf.math.greater_equal_4/GreaterEqual
tf.math.floormod_3/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2
tf.math.floormod_3/FloorMod/yÆ
tf.math.floormod_3/FloorModFloorMod$tf.clip_by_value_4/clip_by_value:z:0&tf.math.floormod_3/FloorMod/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.floormod_3/FloorModµ
$embedding_11/StatefulPartitionedCallStatefulPartitionedCall$tf.clip_by_value_4/clip_by_value:z:0embedding_11_4550*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_embedding_11_layer_call_and_return_conditional_losses_45412&
$embedding_11/StatefulPartitionedCall²
#embedding_9/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_3/FloorDiv:z:0embedding_9_4572*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_embedding_9_layer_call_and_return_conditional_losses_45632%
#embedding_9/StatefulPartitionedCall
tf.cast_4/CastCast(tf.math.greater_equal_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.cast_4/CastÙ
tf.__operators__.add_8/AddV2AddV2-embedding_11/StatefulPartitionedCall:output:0,embedding_9/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_8/AddV2°
$embedding_10/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_3/FloorMod:z:0embedding_10_4596*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_embedding_10_layer_call_and_return_conditional_losses_45872&
$embedding_10/StatefulPartitionedCallÍ
tf.__operators__.add_9/AddV2AddV2 tf.__operators__.add_8/AddV2:z:0-embedding_10/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_9/AddV2
tf.expand_dims_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
tf.expand_dims_3/ExpandDims/dim¼
tf.expand_dims_3/ExpandDims
ExpandDimstf.cast_4/Cast:y:0(tf.expand_dims_3/ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.expand_dims_3/ExpandDims¶
tf.math.multiply_3/MulMul tf.__operators__.add_9/AddV2:z:0$tf.expand_dims_3/ExpandDims:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.multiply_3/Mul
*tf.math.reduce_sum_3/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_3/Sum/reduction_indices¿
tf.math.reduce_sum_3/SumSumtf.math.multiply_3/Mul:z:03tf.math.reduce_sum_3/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.reduce_sum_3/Sumê
IdentityIdentity!tf.math.reduce_sum_3/Sum:output:0%^embedding_10/StatefulPartitionedCall%^embedding_11/StatefulPartitionedCall$^embedding_9/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: :::2L
$embedding_10/StatefulPartitionedCall$embedding_10/StatefulPartitionedCall2L
$embedding_11/StatefulPartitionedCall$embedding_11/StatefulPartitionedCall2J
#embedding_9/StatefulPartitionedCall#embedding_9/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4:

_output_shapes
: 
Í
p
*__inference_embedding_6_layer_call_fn_6523

inputs
unknown
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_embedding_6_layer_call_and_return_conditional_losses_43372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
á
|
'__inference_dense_13_layer_call_fn_6376

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_dense_13_layer_call_and_return_conditional_losses_49332
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´?

__inference__traced_save_6719
file_prefix-
)savev2_dense_9_kernel_read_readvariableop+
'savev2_dense_9_bias_read_readvariableop.
*savev2_dense_10_kernel_read_readvariableop,
(savev2_dense_10_bias_read_readvariableop.
*savev2_dense_12_kernel_read_readvariableop,
(savev2_dense_12_bias_read_readvariableop.
*savev2_dense_11_kernel_read_readvariableop,
(savev2_dense_11_bias_read_readvariableop.
*savev2_dense_13_kernel_read_readvariableop,
(savev2_dense_13_bias_read_readvariableop.
*savev2_dense_14_kernel_read_readvariableop,
(savev2_dense_14_bias_read_readvariableop.
*savev2_dense_15_kernel_read_readvariableop,
(savev2_dense_15_bias_read_readvariableop.
*savev2_dense_16_kernel_read_readvariableop,
(savev2_dense_16_bias_read_readvariableop.
*savev2_dense_17_kernel_read_readvariableop,
(savev2_dense_17_bias_read_readvariableop5
1savev2_embedding_8_embeddings_read_readvariableop5
1savev2_embedding_6_embeddings_read_readvariableop5
1savev2_embedding_7_embeddings_read_readvariableop6
2savev2_embedding_11_embeddings_read_readvariableop5
1savev2_embedding_9_embeddings_read_readvariableop6
2savev2_embedding_10_embeddings_read_readvariableop?
;savev2_normalize_1_normalization_1_mean_read_readvariableopC
?savev2_normalize_1_normalization_1_variance_read_readvariableop@
<savev2_normalize_1_normalization_1_count_read_readvariableop	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const_5

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*¦
valueBB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesÄ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop*savev2_dense_10_kernel_read_readvariableop(savev2_dense_10_bias_read_readvariableop*savev2_dense_12_kernel_read_readvariableop(savev2_dense_12_bias_read_readvariableop*savev2_dense_11_kernel_read_readvariableop(savev2_dense_11_bias_read_readvariableop*savev2_dense_13_kernel_read_readvariableop(savev2_dense_13_bias_read_readvariableop*savev2_dense_14_kernel_read_readvariableop(savev2_dense_14_bias_read_readvariableop*savev2_dense_15_kernel_read_readvariableop(savev2_dense_15_bias_read_readvariableop*savev2_dense_16_kernel_read_readvariableop(savev2_dense_16_bias_read_readvariableop*savev2_dense_17_kernel_read_readvariableop(savev2_dense_17_bias_read_readvariableop1savev2_embedding_8_embeddings_read_readvariableop1savev2_embedding_6_embeddings_read_readvariableop1savev2_embedding_7_embeddings_read_readvariableop2savev2_embedding_11_embeddings_read_readvariableop1savev2_embedding_9_embeddings_read_readvariableop2savev2_embedding_10_embeddings_read_readvariableop;savev2_normalize_1_normalization_1_mean_read_readvariableop?savev2_normalize_1_normalization_1_variance_read_readvariableop<savev2_normalize_1_normalization_1_count_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const_5"/device:CPU:0*
_output_shapes
 *,
dtypes"
 2	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: :
::
::	::
::
::
::
::
::	::	4:	:	:	4:	:	::: : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&	"
 
_output_shapes
:
:!


_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::%!

_output_shapes
:	4:%!

_output_shapes
:	:%!

_output_shapes
:	:%!

_output_shapes
:	4:%!

_output_shapes
:	:%!

_output_shapes
:	:!

_output_shapes	
::!

_output_shapes	
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ï
q
+__inference_embedding_10_layer_call_fn_6602

inputs
unknown
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_embedding_10_layer_call_and_return_conditional_losses_45872
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
-
Á
A__inference_model_2_layer_call_and_return_conditional_losses_4380
input_3*
&tf_math_greater_equal_3_greaterequal_y
embedding_8_4324
embedding_6_4346
embedding_7_4370
identity¢#embedding_6/StatefulPartitionedCall¢#embedding_7/StatefulPartitionedCall¢#embedding_8/StatefulPartitionedCallÛ
flatten_2/PartitionedCallPartitionedCallinput_3*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_flatten_2_layer_call_and_return_conditional_losses_42872
flatten_2/PartitionedCall
*tf.clip_by_value_3/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_3/clip_by_value/Minimum/yê
(tf.clip_by_value_3/clip_by_value/MinimumMinimum"flatten_2/PartitionedCall:output:03tf.clip_by_value_3/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(tf.clip_by_value_3/clip_by_value/Minimum
"tf.clip_by_value_3/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_3/clip_by_value/yÜ
 tf.clip_by_value_3/clip_by_valueMaximum,tf.clip_by_value_3/clip_by_value/Minimum:z:0+tf.clip_by_value_3/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 tf.clip_by_value_3/clip_by_value
#tf.compat.v1.floor_div_2/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2%
#tf.compat.v1.floor_div_2/FloorDiv/yØ
!tf.compat.v1.floor_div_2/FloorDivFloorDiv$tf.clip_by_value_3/clip_by_value:z:0,tf.compat.v1.floor_div_2/FloorDiv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!tf.compat.v1.floor_div_2/FloorDivÚ
$tf.math.greater_equal_3/GreaterEqualGreaterEqual"flatten_2/PartitionedCall:output:0&tf_math_greater_equal_3_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$tf.math.greater_equal_3/GreaterEqual
tf.math.floormod_2/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2
tf.math.floormod_2/FloorMod/yÆ
tf.math.floormod_2/FloorModFloorMod$tf.clip_by_value_3/clip_by_value:z:0&tf.math.floormod_2/FloorMod/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.floormod_2/FloorMod±
#embedding_8/StatefulPartitionedCallStatefulPartitionedCall$tf.clip_by_value_3/clip_by_value:z:0embedding_8_4324*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_embedding_8_layer_call_and_return_conditional_losses_43152%
#embedding_8/StatefulPartitionedCall²
#embedding_6/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_2/FloorDiv:z:0embedding_6_4346*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_embedding_6_layer_call_and_return_conditional_losses_43372%
#embedding_6/StatefulPartitionedCall
tf.cast_3/CastCast(tf.math.greater_equal_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.cast_3/CastØ
tf.__operators__.add_6/AddV2AddV2,embedding_8/StatefulPartitionedCall:output:0,embedding_6/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_6/AddV2¬
#embedding_7/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_2/FloorMod:z:0embedding_7_4370*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_embedding_7_layer_call_and_return_conditional_losses_43612%
#embedding_7/StatefulPartitionedCallÌ
tf.__operators__.add_7/AddV2AddV2 tf.__operators__.add_6/AddV2:z:0,embedding_7/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_7/AddV2
tf.expand_dims_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
tf.expand_dims_2/ExpandDims/dim¼
tf.expand_dims_2/ExpandDims
ExpandDimstf.cast_3/Cast:y:0(tf.expand_dims_2/ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.expand_dims_2/ExpandDims¶
tf.math.multiply_2/MulMul tf.__operators__.add_7/AddV2:z:0$tf.expand_dims_2/ExpandDims:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.multiply_2/Mul
*tf.math.reduce_sum_2/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_2/Sum/reduction_indices¿
tf.math.reduce_sum_2/SumSumtf.math.multiply_2/Mul:z:03tf.math.reduce_sum_2/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.reduce_sum_2/Sumè
IdentityIdentity!tf.math.reduce_sum_2/Sum:output:0$^embedding_6/StatefulPartitionedCall$^embedding_7/StatefulPartitionedCall$^embedding_8/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: :::2J
#embedding_6/StatefulPartitionedCall#embedding_6/StatefulPartitionedCall2J
#embedding_7/StatefulPartitionedCall#embedding_7/StatefulPartitionedCall2J
#embedding_8/StatefulPartitionedCall#embedding_8/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_3:

_output_shapes
: 
Í
p
*__inference_embedding_8_layer_call_fn_6506

inputs
unknown
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_embedding_8_layer_call_and_return_conditional_losses_43152
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¹

&__inference_model_3_layer_call_fn_6265

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_model_3_layer_call_and_return_conditional_losses_46732
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 
¬V
µ	
H__inference_custom_model_1_layer_call_and_return_conditional_losses_5094

cards0

cards1
bets*
&tf_math_greater_equal_5_greaterequal_y
model_2_4763
model_2_4765
model_2_4767
model_2_4769
model_3_4798
model_3_4800
model_3_4802
model_3_4804.
*tf_clip_by_value_5_clip_by_value_minimum_y&
"tf_clip_by_value_5_clip_by_value_y
dense_9_4838
dense_9_4840
dense_12_4864
dense_12_4866
dense_10_4891
dense_10_4893
dense_11_4918
dense_11_4920
dense_13_4944
dense_13_4946
dense_14_4972
dense_14_4974
dense_15_4999
dense_15_5001
dense_16_5027
dense_16_5029
normalize_1_5062
normalize_1_5064
dense_17_5088
dense_17_5090
identity¢ dense_10/StatefulPartitionedCall¢ dense_11/StatefulPartitionedCall¢ dense_12/StatefulPartitionedCall¢ dense_13/StatefulPartitionedCall¢ dense_14/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall¢ dense_16/StatefulPartitionedCall¢ dense_17/StatefulPartitionedCall¢dense_9/StatefulPartitionedCall¢model_2/StatefulPartitionedCall¢model_3/StatefulPartitionedCall¢#normalize_1/StatefulPartitionedCall¼
$tf.math.greater_equal_5/GreaterEqualGreaterEqualbets&tf_math_greater_equal_5_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2&
$tf.math.greater_equal_5/GreaterEqual®
model_2/StatefulPartitionedCallStatefulPartitionedCallcards0model_2_4763model_2_4765model_2_4767model_2_4769*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_model_2_layer_call_and_return_conditional_losses_44472!
model_2/StatefulPartitionedCall®
model_3/StatefulPartitionedCallStatefulPartitionedCallcards1model_3_4798model_3_4800model_3_4802model_3_4804*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_model_3_layer_call_and_return_conditional_losses_46732!
model_3/StatefulPartitionedCallÃ
(tf.clip_by_value_5/clip_by_value/MinimumMinimumbets*tf_clip_by_value_5_clip_by_value_minimum_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2*
(tf.clip_by_value_5/clip_by_value/MinimumÓ
 tf.clip_by_value_5/clip_by_valueMaximum,tf.clip_by_value_5/clip_by_value/Minimum:z:0"tf_clip_by_value_5_clip_by_value_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2"
 tf.clip_by_value_5/clip_by_value
tf.cast_5/CastCast(tf.math.greater_equal_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
tf.cast_5/Castt
tf.concat_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_3/concat/axisæ
tf.concat_3/concatConcatV2(model_2/StatefulPartitionedCall:output:0(model_3/StatefulPartitionedCall:output:0 tf.concat_3/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.concat_3/concat}
tf.concat_4/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
tf.concat_4/concat/axisË
tf.concat_4/concatConcatV2$tf.clip_by_value_5/clip_by_value:z:0tf.cast_5/Cast:y:0 tf.concat_4/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.concat_4/concat¤
dense_9/StatefulPartitionedCallStatefulPartitionedCalltf.concat_3/concat:output:0dense_9_4838dense_9_4840*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_dense_9_layer_call_and_return_conditional_losses_48272!
dense_9/StatefulPartitionedCall©
 dense_12/StatefulPartitionedCallStatefulPartitionedCalltf.concat_4/concat:output:0dense_12_4864dense_12_4866*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_dense_12_layer_call_and_return_conditional_losses_48532"
 dense_12/StatefulPartitionedCall¶
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0dense_10_4891dense_10_4893*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_dense_10_layer_call_and_return_conditional_losses_48802"
 dense_10/StatefulPartitionedCall·
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_4918dense_11_4920*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_dense_11_layer_call_and_return_conditional_losses_49072"
 dense_11/StatefulPartitionedCall·
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_4944dense_13_4946*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_dense_13_layer_call_and_return_conditional_losses_49332"
 dense_13/StatefulPartitionedCall}
tf.concat_5/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
tf.concat_5/concat/axisè
tf.concat_5/concatConcatV2)dense_11/StatefulPartitionedCall:output:0)dense_13/StatefulPartitionedCall:output:0 tf.concat_5/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.concat_5/concat©
 dense_14/StatefulPartitionedCallStatefulPartitionedCalltf.concat_5/concat:output:0dense_14_4972dense_14_4974*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_dense_14_layer_call_and_return_conditional_losses_49612"
 dense_14/StatefulPartitionedCall
tf.nn.relu_3/ReluRelu)dense_14/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.nn.relu_3/Relu­
 dense_15/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_3/Relu:activations:0dense_15_4999dense_15_5001*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_dense_15_layer_call_and_return_conditional_losses_49882"
 dense_15/StatefulPartitionedCallÆ
tf.__operators__.add_10/AddV2AddV2)dense_15/StatefulPartitionedCall:output:0tf.nn.relu_3/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_10/AddV2
tf.nn.relu_4/ReluRelu!tf.__operators__.add_10/AddV2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.nn.relu_4/Relu­
 dense_16/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_4/Relu:activations:0dense_16_5027dense_16_5029*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_dense_16_layer_call_and_return_conditional_losses_50162"
 dense_16/StatefulPartitionedCallÆ
tf.__operators__.add_11/AddV2AddV2)dense_16/StatefulPartitionedCall:output:0tf.nn.relu_4/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_11/AddV2
tf.nn.relu_5/ReluRelu!tf.__operators__.add_11/AddV2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.nn.relu_5/Relu¼
#normalize_1/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_5/Relu:activations:0normalize_1_5062normalize_1_5064*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_normalize_1_layer_call_and_return_conditional_losses_50512%
#normalize_1/StatefulPartitionedCall¹
 dense_17/StatefulPartitionedCallStatefulPartitionedCall,normalize_1/StatefulPartitionedCall:output:0dense_17_5088dense_17_5090*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_dense_17_layer_call_and_return_conditional_losses_50772"
 dense_17/StatefulPartitionedCall¡
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall ^model_2/StatefulPartitionedCall ^model_3/StatefulPartitionedCall$^normalize_1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*À
_input_shapes®
«:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
: : :::: :::: : ::::::::::::::::::::2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2B
model_2/StatefulPartitionedCallmodel_2/StatefulPartitionedCall2B
model_3/StatefulPartitionedCallmodel_3/StatefulPartitionedCall2J
#normalize_1/StatefulPartitionedCall#normalize_1/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namecards0:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namecards1:MI
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


_user_specified_namebets:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
¨
ß
-__inference_custom_model_1_layer_call_fn_5509

cards0

cards1
bets
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallcards0cards1betsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
 !*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_custom_model_1_layer_call_and_return_conditional_losses_54442
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*À
_input_shapes®
«:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
: : :::: :::: : ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namecards0:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namecards1:MI
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


_user_specified_namebets:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
	
Û
B__inference_dense_13_layer_call_and_return_conditional_losses_6367

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù	

E__inference_embedding_8_layer_call_and_return_conditional_losses_6499

inputs
embedding_lookup_6493
identity¢embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Castú
embedding_lookupResourceGatherembedding_lookup_6493Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*(
_class
loc:@embedding_lookup/6493*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_lookupì
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@embedding_lookup/6493*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity¡
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
©
_
C__inference_flatten_3_layer_call_and_return_conditional_losses_4513

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼

&__inference_model_3_layer_call_fn_4729
input_4
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_model_3_layer_call_and_return_conditional_losses_47182
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: :::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4:

_output_shapes
: 
Ú	

F__inference_embedding_10_layer_call_and_return_conditional_losses_6595

inputs
embedding_lookup_6589
identity¢embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Castú
embedding_lookupResourceGatherembedding_lookup_6589Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*(
_class
loc:@embedding_lookup/6589*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_lookupì
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@embedding_lookup/6589*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity¡
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ú	

F__inference_embedding_11_layer_call_and_return_conditional_losses_4541

inputs
embedding_lookup_4535
identity¢embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Castú
embedding_lookupResourceGatherembedding_lookup_4535Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*(
_class
loc:@embedding_lookup/4535*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_lookupì
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@embedding_lookup/4535*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity¡
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù	

E__inference_embedding_9_layer_call_and_return_conditional_losses_4563

inputs
embedding_lookup_4557
identity¢embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Castú
embedding_lookupResourceGatherembedding_lookup_4557Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*(
_class
loc:@embedding_lookup/4557*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_lookupì
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@embedding_lookup/4557*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity¡
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù	

E__inference_embedding_7_layer_call_and_return_conditional_losses_4361

inputs
embedding_lookup_4355
identity¢embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Castú
embedding_lookupResourceGatherembedding_lookup_4355Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*(
_class
loc:@embedding_lookup/4355*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_lookupì
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@embedding_lookup/4355*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity¡
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
-
Á
A__inference_model_2_layer_call_and_return_conditional_losses_4412
input_3*
&tf_math_greater_equal_3_greaterequal_y
embedding_8_4394
embedding_6_4397
embedding_7_4402
identity¢#embedding_6/StatefulPartitionedCall¢#embedding_7/StatefulPartitionedCall¢#embedding_8/StatefulPartitionedCallÛ
flatten_2/PartitionedCallPartitionedCallinput_3*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_flatten_2_layer_call_and_return_conditional_losses_42872
flatten_2/PartitionedCall
*tf.clip_by_value_3/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_3/clip_by_value/Minimum/yê
(tf.clip_by_value_3/clip_by_value/MinimumMinimum"flatten_2/PartitionedCall:output:03tf.clip_by_value_3/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(tf.clip_by_value_3/clip_by_value/Minimum
"tf.clip_by_value_3/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_3/clip_by_value/yÜ
 tf.clip_by_value_3/clip_by_valueMaximum,tf.clip_by_value_3/clip_by_value/Minimum:z:0+tf.clip_by_value_3/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 tf.clip_by_value_3/clip_by_value
#tf.compat.v1.floor_div_2/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2%
#tf.compat.v1.floor_div_2/FloorDiv/yØ
!tf.compat.v1.floor_div_2/FloorDivFloorDiv$tf.clip_by_value_3/clip_by_value:z:0,tf.compat.v1.floor_div_2/FloorDiv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!tf.compat.v1.floor_div_2/FloorDivÚ
$tf.math.greater_equal_3/GreaterEqualGreaterEqual"flatten_2/PartitionedCall:output:0&tf_math_greater_equal_3_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$tf.math.greater_equal_3/GreaterEqual
tf.math.floormod_2/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2
tf.math.floormod_2/FloorMod/yÆ
tf.math.floormod_2/FloorModFloorMod$tf.clip_by_value_3/clip_by_value:z:0&tf.math.floormod_2/FloorMod/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.floormod_2/FloorMod±
#embedding_8/StatefulPartitionedCallStatefulPartitionedCall$tf.clip_by_value_3/clip_by_value:z:0embedding_8_4394*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_embedding_8_layer_call_and_return_conditional_losses_43152%
#embedding_8/StatefulPartitionedCall²
#embedding_6/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_2/FloorDiv:z:0embedding_6_4397*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_embedding_6_layer_call_and_return_conditional_losses_43372%
#embedding_6/StatefulPartitionedCall
tf.cast_3/CastCast(tf.math.greater_equal_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.cast_3/CastØ
tf.__operators__.add_6/AddV2AddV2,embedding_8/StatefulPartitionedCall:output:0,embedding_6/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_6/AddV2¬
#embedding_7/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_2/FloorMod:z:0embedding_7_4402*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_embedding_7_layer_call_and_return_conditional_losses_43612%
#embedding_7/StatefulPartitionedCallÌ
tf.__operators__.add_7/AddV2AddV2 tf.__operators__.add_6/AddV2:z:0,embedding_7/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_7/AddV2
tf.expand_dims_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
tf.expand_dims_2/ExpandDims/dim¼
tf.expand_dims_2/ExpandDims
ExpandDimstf.cast_3/Cast:y:0(tf.expand_dims_2/ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.expand_dims_2/ExpandDims¶
tf.math.multiply_2/MulMul tf.__operators__.add_7/AddV2:z:0$tf.expand_dims_2/ExpandDims:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.multiply_2/Mul
*tf.math.reduce_sum_2/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_2/Sum/reduction_indices¿
tf.math.reduce_sum_2/SumSumtf.math.multiply_2/Mul:z:03tf.math.reduce_sum_2/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.reduce_sum_2/Sumè
IdentityIdentity!tf.math.reduce_sum_2/Sum:output:0$^embedding_6/StatefulPartitionedCall$^embedding_7/StatefulPartitionedCall$^embedding_8/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: :::2J
#embedding_6/StatefulPartitionedCall#embedding_6/StatefulPartitionedCall2J
#embedding_7/StatefulPartitionedCall#embedding_7/StatefulPartitionedCall2J
#embedding_8/StatefulPartitionedCall#embedding_8/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_3:

_output_shapes
: 
ô	
Ú
A__inference_dense_9_layer_call_and_return_conditional_losses_4827

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ç

E__inference_normalize_1_layer_call_and_return_conditional_losses_5051
x3
/normalization_1_reshape_readvariableop_resource5
1normalization_1_reshape_1_readvariableop_resource
identity¢&normalization_1/Reshape/ReadVariableOp¢(normalization_1/Reshape_1/ReadVariableOp½
&normalization_1/Reshape/ReadVariableOpReadVariableOp/normalization_1_reshape_readvariableop_resource*
_output_shapes	
:*
dtype02(
&normalization_1/Reshape/ReadVariableOp
normalization_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_1/Reshape/shape¿
normalization_1/ReshapeReshape.normalization_1/Reshape/ReadVariableOp:value:0&normalization_1/Reshape/shape:output:0*
T0*
_output_shapes
:	2
normalization_1/ReshapeÃ
(normalization_1/Reshape_1/ReadVariableOpReadVariableOp1normalization_1_reshape_1_readvariableop_resource*
_output_shapes	
:*
dtype02*
(normalization_1/Reshape_1/ReadVariableOp
normalization_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_1/Reshape_1/shapeÇ
normalization_1/Reshape_1Reshape0normalization_1/Reshape_1/ReadVariableOp:value:0(normalization_1/Reshape_1/shape:output:0*
T0*
_output_shapes
:	2
normalization_1/Reshape_1
normalization_1/subSubx normalization_1/Reshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
normalization_1/sub
normalization_1/SqrtSqrt"normalization_1/Reshape_1:output:0*
T0*
_output_shapes
:	2
normalization_1/Sqrt{
normalization_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö32
normalization_1/Maximum/y¥
normalization_1/MaximumMaximumnormalization_1/Sqrt:y:0"normalization_1/Maximum/y:output:0*
T0*
_output_shapes
:	2
normalization_1/Maximum¦
normalization_1/truedivRealDivnormalization_1/sub:z:0normalization_1/Maximum:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
normalization_1/truedivÄ
IdentityIdentitynormalization_1/truediv:z:0'^normalization_1/Reshape/ReadVariableOp)^normalization_1/Reshape_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::2P
&normalization_1/Reshape/ReadVariableOp&normalization_1/Reshape/ReadVariableOp2T
(normalization_1/Reshape_1/ReadVariableOp(normalization_1/Reshape_1/ReadVariableOp:K G
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
Ì
ë
-__inference_custom_model_1_layer_call_fn_6058

inputs_0_0

inputs_0_1
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29
identity¢StatefulPartitionedCall¡
StatefulPartitionedCallStatefulPartitionedCall
inputs_0_0
inputs_0_1inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
 !*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_custom_model_1_layer_call_and_return_conditional_losses_54442
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*À
_input_shapes®
«:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
: : :::: :::: : ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/0/0:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/0/1:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ô	
Ú
A__inference_dense_9_layer_call_and_return_conditional_losses_6289

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Û
B__inference_dense_15_layer_call_and_return_conditional_losses_6405

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾V
»	
H__inference_custom_model_1_layer_call_and_return_conditional_losses_5444

inputs
inputs_1
inputs_2*
&tf_math_greater_equal_5_greaterequal_y
model_2_5359
model_2_5361
model_2_5363
model_2_5365
model_3_5368
model_3_5370
model_3_5372
model_3_5374.
*tf_clip_by_value_5_clip_by_value_minimum_y&
"tf_clip_by_value_5_clip_by_value_y
dense_9_5386
dense_9_5388
dense_12_5391
dense_12_5393
dense_10_5396
dense_10_5398
dense_11_5401
dense_11_5403
dense_13_5406
dense_13_5408
dense_14_5413
dense_14_5415
dense_15_5419
dense_15_5421
dense_16_5426
dense_16_5428
normalize_1_5433
normalize_1_5435
dense_17_5438
dense_17_5440
identity¢ dense_10/StatefulPartitionedCall¢ dense_11/StatefulPartitionedCall¢ dense_12/StatefulPartitionedCall¢ dense_13/StatefulPartitionedCall¢ dense_14/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall¢ dense_16/StatefulPartitionedCall¢ dense_17/StatefulPartitionedCall¢dense_9/StatefulPartitionedCall¢model_2/StatefulPartitionedCall¢model_3/StatefulPartitionedCall¢#normalize_1/StatefulPartitionedCallÀ
$tf.math.greater_equal_5/GreaterEqualGreaterEqualinputs_2&tf_math_greater_equal_5_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2&
$tf.math.greater_equal_5/GreaterEqual®
model_2/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_2_5359model_2_5361model_2_5363model_2_5365*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_model_2_layer_call_and_return_conditional_losses_44922!
model_2/StatefulPartitionedCall°
model_3/StatefulPartitionedCallStatefulPartitionedCallinputs_1model_3_5368model_3_5370model_3_5372model_3_5374*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_model_3_layer_call_and_return_conditional_losses_47182!
model_3/StatefulPartitionedCallÇ
(tf.clip_by_value_5/clip_by_value/MinimumMinimuminputs_2*tf_clip_by_value_5_clip_by_value_minimum_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2*
(tf.clip_by_value_5/clip_by_value/MinimumÓ
 tf.clip_by_value_5/clip_by_valueMaximum,tf.clip_by_value_5/clip_by_value/Minimum:z:0"tf_clip_by_value_5_clip_by_value_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2"
 tf.clip_by_value_5/clip_by_value
tf.cast_5/CastCast(tf.math.greater_equal_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
tf.cast_5/Castt
tf.concat_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_3/concat/axisæ
tf.concat_3/concatConcatV2(model_2/StatefulPartitionedCall:output:0(model_3/StatefulPartitionedCall:output:0 tf.concat_3/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.concat_3/concat}
tf.concat_4/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
tf.concat_4/concat/axisË
tf.concat_4/concatConcatV2$tf.clip_by_value_5/clip_by_value:z:0tf.cast_5/Cast:y:0 tf.concat_4/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.concat_4/concat¤
dense_9/StatefulPartitionedCallStatefulPartitionedCalltf.concat_3/concat:output:0dense_9_5386dense_9_5388*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_dense_9_layer_call_and_return_conditional_losses_48272!
dense_9/StatefulPartitionedCall©
 dense_12/StatefulPartitionedCallStatefulPartitionedCalltf.concat_4/concat:output:0dense_12_5391dense_12_5393*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_dense_12_layer_call_and_return_conditional_losses_48532"
 dense_12/StatefulPartitionedCall¶
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0dense_10_5396dense_10_5398*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_dense_10_layer_call_and_return_conditional_losses_48802"
 dense_10/StatefulPartitionedCall·
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_5401dense_11_5403*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_dense_11_layer_call_and_return_conditional_losses_49072"
 dense_11/StatefulPartitionedCall·
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_5406dense_13_5408*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_dense_13_layer_call_and_return_conditional_losses_49332"
 dense_13/StatefulPartitionedCall}
tf.concat_5/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
tf.concat_5/concat/axisè
tf.concat_5/concatConcatV2)dense_11/StatefulPartitionedCall:output:0)dense_13/StatefulPartitionedCall:output:0 tf.concat_5/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.concat_5/concat©
 dense_14/StatefulPartitionedCallStatefulPartitionedCalltf.concat_5/concat:output:0dense_14_5413dense_14_5415*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_dense_14_layer_call_and_return_conditional_losses_49612"
 dense_14/StatefulPartitionedCall
tf.nn.relu_3/ReluRelu)dense_14/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.nn.relu_3/Relu­
 dense_15/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_3/Relu:activations:0dense_15_5419dense_15_5421*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_dense_15_layer_call_and_return_conditional_losses_49882"
 dense_15/StatefulPartitionedCallÆ
tf.__operators__.add_10/AddV2AddV2)dense_15/StatefulPartitionedCall:output:0tf.nn.relu_3/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_10/AddV2
tf.nn.relu_4/ReluRelu!tf.__operators__.add_10/AddV2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.nn.relu_4/Relu­
 dense_16/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_4/Relu:activations:0dense_16_5426dense_16_5428*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_dense_16_layer_call_and_return_conditional_losses_50162"
 dense_16/StatefulPartitionedCallÆ
tf.__operators__.add_11/AddV2AddV2)dense_16/StatefulPartitionedCall:output:0tf.nn.relu_4/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_11/AddV2
tf.nn.relu_5/ReluRelu!tf.__operators__.add_11/AddV2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.nn.relu_5/Relu¼
#normalize_1/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_5/Relu:activations:0normalize_1_5433normalize_1_5435*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_normalize_1_layer_call_and_return_conditional_losses_50512%
#normalize_1/StatefulPartitionedCall¹
 dense_17/StatefulPartitionedCallStatefulPartitionedCall,normalize_1/StatefulPartitionedCall:output:0dense_17_5438dense_17_5440*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_dense_17_layer_call_and_return_conditional_losses_50772"
 dense_17/StatefulPartitionedCall¡
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall ^model_2/StatefulPartitionedCall ^model_3/StatefulPartitionedCall$^normalize_1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*À
_input_shapes®
«:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
: : :::: :::: : ::::::::::::::::::::2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2B
model_2/StatefulPartitionedCallmodel_2/StatefulPartitionedCall2B
model_3/StatefulPartitionedCallmodel_3/StatefulPartitionedCall2J
#normalize_1/StatefulPartitionedCall#normalize_1/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ú	

F__inference_embedding_11_layer_call_and_return_conditional_losses_6561

inputs
embedding_lookup_6555
identity¢embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Castú
embedding_lookupResourceGatherembedding_lookup_6555Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*(
_class
loc:@embedding_lookup/6555*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_lookupì
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@embedding_lookup/6555*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity¡
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Í
p
*__inference_embedding_7_layer_call_fn_6540

inputs
unknown
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_embedding_7_layer_call_and_return_conditional_losses_43612
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
-
À
A__inference_model_2_layer_call_and_return_conditional_losses_4447

inputs*
&tf_math_greater_equal_3_greaterequal_y
embedding_8_4429
embedding_6_4432
embedding_7_4437
identity¢#embedding_6/StatefulPartitionedCall¢#embedding_7/StatefulPartitionedCall¢#embedding_8/StatefulPartitionedCallÚ
flatten_2/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_flatten_2_layer_call_and_return_conditional_losses_42872
flatten_2/PartitionedCall
*tf.clip_by_value_3/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_3/clip_by_value/Minimum/yê
(tf.clip_by_value_3/clip_by_value/MinimumMinimum"flatten_2/PartitionedCall:output:03tf.clip_by_value_3/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(tf.clip_by_value_3/clip_by_value/Minimum
"tf.clip_by_value_3/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_3/clip_by_value/yÜ
 tf.clip_by_value_3/clip_by_valueMaximum,tf.clip_by_value_3/clip_by_value/Minimum:z:0+tf.clip_by_value_3/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 tf.clip_by_value_3/clip_by_value
#tf.compat.v1.floor_div_2/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2%
#tf.compat.v1.floor_div_2/FloorDiv/yØ
!tf.compat.v1.floor_div_2/FloorDivFloorDiv$tf.clip_by_value_3/clip_by_value:z:0,tf.compat.v1.floor_div_2/FloorDiv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!tf.compat.v1.floor_div_2/FloorDivÚ
$tf.math.greater_equal_3/GreaterEqualGreaterEqual"flatten_2/PartitionedCall:output:0&tf_math_greater_equal_3_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$tf.math.greater_equal_3/GreaterEqual
tf.math.floormod_2/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2
tf.math.floormod_2/FloorMod/yÆ
tf.math.floormod_2/FloorModFloorMod$tf.clip_by_value_3/clip_by_value:z:0&tf.math.floormod_2/FloorMod/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.floormod_2/FloorMod±
#embedding_8/StatefulPartitionedCallStatefulPartitionedCall$tf.clip_by_value_3/clip_by_value:z:0embedding_8_4429*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_embedding_8_layer_call_and_return_conditional_losses_43152%
#embedding_8/StatefulPartitionedCall²
#embedding_6/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_2/FloorDiv:z:0embedding_6_4432*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_embedding_6_layer_call_and_return_conditional_losses_43372%
#embedding_6/StatefulPartitionedCall
tf.cast_3/CastCast(tf.math.greater_equal_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.cast_3/CastØ
tf.__operators__.add_6/AddV2AddV2,embedding_8/StatefulPartitionedCall:output:0,embedding_6/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_6/AddV2¬
#embedding_7/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_2/FloorMod:z:0embedding_7_4437*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_embedding_7_layer_call_and_return_conditional_losses_43612%
#embedding_7/StatefulPartitionedCallÌ
tf.__operators__.add_7/AddV2AddV2 tf.__operators__.add_6/AddV2:z:0,embedding_7/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_7/AddV2
tf.expand_dims_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
tf.expand_dims_2/ExpandDims/dim¼
tf.expand_dims_2/ExpandDims
ExpandDimstf.cast_3/Cast:y:0(tf.expand_dims_2/ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.expand_dims_2/ExpandDims¶
tf.math.multiply_2/MulMul tf.__operators__.add_7/AddV2:z:0$tf.expand_dims_2/ExpandDims:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.multiply_2/Mul
*tf.math.reduce_sum_2/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_2/Sum/reduction_indices¿
tf.math.reduce_sum_2/SumSumtf.math.multiply_2/Mul:z:03tf.math.reduce_sum_2/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.reduce_sum_2/Sumè
IdentityIdentity!tf.math.reduce_sum_2/Sum:output:0$^embedding_6/StatefulPartitionedCall$^embedding_7/StatefulPartitionedCall$^embedding_8/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: :::2J
#embedding_6/StatefulPartitionedCall#embedding_6/StatefulPartitionedCall2J
#embedding_7/StatefulPartitionedCall#embedding_7/StatefulPartitionedCall2J
#embedding_8/StatefulPartitionedCall#embedding_8/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 
Í
p
*__inference_embedding_9_layer_call_fn_6585

inputs
unknown
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_embedding_9_layer_call_and_return_conditional_losses_45632
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
õ	
Û
B__inference_dense_11_layer_call_and_return_conditional_losses_6348

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°8
â
A__inference_model_3_layer_call_and_return_conditional_losses_6252

inputs*
&tf_math_greater_equal_4_greaterequal_y&
"embedding_11_embedding_lookup_6226%
!embedding_9_embedding_lookup_6232&
"embedding_10_embedding_lookup_6240
identity¢embedding_10/embedding_lookup¢embedding_11/embedding_lookup¢embedding_9/embedding_lookups
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten_3/Const
flatten_3/ReshapeReshapeinputsflatten_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten_3/Reshape
*tf.clip_by_value_4/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_4/clip_by_value/Minimum/yâ
(tf.clip_by_value_4/clip_by_value/MinimumMinimumflatten_3/Reshape:output:03tf.clip_by_value_4/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(tf.clip_by_value_4/clip_by_value/Minimum
"tf.clip_by_value_4/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_4/clip_by_value/yÜ
 tf.clip_by_value_4/clip_by_valueMaximum,tf.clip_by_value_4/clip_by_value/Minimum:z:0+tf.clip_by_value_4/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 tf.clip_by_value_4/clip_by_value
#tf.compat.v1.floor_div_3/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2%
#tf.compat.v1.floor_div_3/FloorDiv/yØ
!tf.compat.v1.floor_div_3/FloorDivFloorDiv$tf.clip_by_value_4/clip_by_value:z:0,tf.compat.v1.floor_div_3/FloorDiv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!tf.compat.v1.floor_div_3/FloorDivÒ
$tf.math.greater_equal_4/GreaterEqualGreaterEqualflatten_3/Reshape:output:0&tf_math_greater_equal_4_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$tf.math.greater_equal_4/GreaterEqual
tf.math.floormod_3/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2
tf.math.floormod_3/FloorMod/yÆ
tf.math.floormod_3/FloorModFloorMod$tf.clip_by_value_4/clip_by_value:z:0&tf.math.floormod_3/FloorMod/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.floormod_3/FloorMod
embedding_11/CastCast$tf.clip_by_value_4/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_11/Cast»
embedding_11/embedding_lookupResourceGather"embedding_11_embedding_lookup_6226embedding_11/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*5
_class+
)'loc:@embedding_11/embedding_lookup/6226*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_11/embedding_lookup 
&embedding_11/embedding_lookup/IdentityIdentity&embedding_11/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*5
_class+
)'loc:@embedding_11/embedding_lookup/6226*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&embedding_11/embedding_lookup/IdentityÈ
(embedding_11/embedding_lookup/Identity_1Identity/embedding_11/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(embedding_11/embedding_lookup/Identity_1
embedding_9/CastCast%tf.compat.v1.floor_div_3/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_9/Cast¶
embedding_9/embedding_lookupResourceGather!embedding_9_embedding_lookup_6232embedding_9/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*4
_class*
(&loc:@embedding_9/embedding_lookup/6232*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_9/embedding_lookup
%embedding_9/embedding_lookup/IdentityIdentity%embedding_9/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*4
_class*
(&loc:@embedding_9/embedding_lookup/6232*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%embedding_9/embedding_lookup/IdentityÅ
'embedding_9/embedding_lookup/Identity_1Identity.embedding_9/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'embedding_9/embedding_lookup/Identity_1
tf.cast_4/CastCast(tf.math.greater_equal_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.cast_4/Castá
tf.__operators__.add_8/AddV2AddV21embedding_11/embedding_lookup/Identity_1:output:00embedding_9/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_8/AddV2
embedding_10/CastCasttf.math.floormod_3/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_10/Cast»
embedding_10/embedding_lookupResourceGather"embedding_10_embedding_lookup_6240embedding_10/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*5
_class+
)'loc:@embedding_10/embedding_lookup/6240*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_10/embedding_lookup 
&embedding_10/embedding_lookup/IdentityIdentity&embedding_10/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*5
_class+
)'loc:@embedding_10/embedding_lookup/6240*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&embedding_10/embedding_lookup/IdentityÈ
(embedding_10/embedding_lookup/Identity_1Identity/embedding_10/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(embedding_10/embedding_lookup/Identity_1Ñ
tf.__operators__.add_9/AddV2AddV2 tf.__operators__.add_8/AddV2:z:01embedding_10/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_9/AddV2
tf.expand_dims_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
tf.expand_dims_3/ExpandDims/dim¼
tf.expand_dims_3/ExpandDims
ExpandDimstf.cast_4/Cast:y:0(tf.expand_dims_3/ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.expand_dims_3/ExpandDims¶
tf.math.multiply_3/MulMul tf.__operators__.add_9/AddV2:z:0$tf.expand_dims_3/ExpandDims:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.multiply_3/Mul
*tf.math.reduce_sum_3/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_3/Sum/reduction_indices¿
tf.math.reduce_sum_3/SumSumtf.math.multiply_3/Mul:z:03tf.math.reduce_sum_3/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.reduce_sum_3/SumÕ
IdentityIdentity!tf.math.reduce_sum_3/Sum:output:0^embedding_10/embedding_lookup^embedding_11/embedding_lookup^embedding_9/embedding_lookup*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: :::2>
embedding_10/embedding_lookupembedding_10/embedding_lookup2>
embedding_11/embedding_lookupembedding_11/embedding_lookup2<
embedding_9/embedding_lookupembedding_9/embedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 
×ö
÷
H__inference_custom_model_1_layer_call_and_return_conditional_losses_5750

inputs_0_0

inputs_0_1
inputs_1*
&tf_math_greater_equal_5_greaterequal_y2
.model_2_tf_math_greater_equal_3_greaterequal_y-
)model_2_embedding_8_embedding_lookup_5600-
)model_2_embedding_6_embedding_lookup_5606-
)model_2_embedding_7_embedding_lookup_56142
.model_3_tf_math_greater_equal_4_greaterequal_y.
*model_3_embedding_11_embedding_lookup_5638-
)model_3_embedding_9_embedding_lookup_5644.
*model_3_embedding_10_embedding_lookup_5652.
*tf_clip_by_value_5_clip_by_value_minimum_y&
"tf_clip_by_value_5_clip_by_value_y*
&dense_9_matmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource+
'dense_10_matmul_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource+
'dense_11_matmul_readvariableop_resource,
(dense_11_biasadd_readvariableop_resource+
'dense_13_matmul_readvariableop_resource,
(dense_13_biasadd_readvariableop_resource+
'dense_14_matmul_readvariableop_resource,
(dense_14_biasadd_readvariableop_resource+
'dense_15_matmul_readvariableop_resource,
(dense_15_biasadd_readvariableop_resource+
'dense_16_matmul_readvariableop_resource,
(dense_16_biasadd_readvariableop_resource?
;normalize_1_normalization_1_reshape_readvariableop_resourceA
=normalize_1_normalization_1_reshape_1_readvariableop_resource+
'dense_17_matmul_readvariableop_resource,
(dense_17_biasadd_readvariableop_resource
identity¢dense_10/BiasAdd/ReadVariableOp¢dense_10/MatMul/ReadVariableOp¢dense_11/BiasAdd/ReadVariableOp¢dense_11/MatMul/ReadVariableOp¢dense_12/BiasAdd/ReadVariableOp¢dense_12/MatMul/ReadVariableOp¢dense_13/BiasAdd/ReadVariableOp¢dense_13/MatMul/ReadVariableOp¢dense_14/BiasAdd/ReadVariableOp¢dense_14/MatMul/ReadVariableOp¢dense_15/BiasAdd/ReadVariableOp¢dense_15/MatMul/ReadVariableOp¢dense_16/BiasAdd/ReadVariableOp¢dense_16/MatMul/ReadVariableOp¢dense_17/BiasAdd/ReadVariableOp¢dense_17/MatMul/ReadVariableOp¢dense_9/BiasAdd/ReadVariableOp¢dense_9/MatMul/ReadVariableOp¢$model_2/embedding_6/embedding_lookup¢$model_2/embedding_7/embedding_lookup¢$model_2/embedding_8/embedding_lookup¢%model_3/embedding_10/embedding_lookup¢%model_3/embedding_11/embedding_lookup¢$model_3/embedding_9/embedding_lookup¢2normalize_1/normalization_1/Reshape/ReadVariableOp¢4normalize_1/normalization_1/Reshape_1/ReadVariableOpÀ
$tf.math.greater_equal_5/GreaterEqualGreaterEqualinputs_1&tf_math_greater_equal_5_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2&
$tf.math.greater_equal_5/GreaterEqual
model_2/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
model_2/flatten_2/Const¡
model_2/flatten_2/ReshapeReshape
inputs_0_0 model_2/flatten_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_2/flatten_2/Reshape­
2model_2/tf.clip_by_value_3/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI24
2model_2/tf.clip_by_value_3/clip_by_value/Minimum/y
0model_2/tf.clip_by_value_3/clip_by_value/MinimumMinimum"model_2/flatten_2/Reshape:output:0;model_2/tf.clip_by_value_3/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0model_2/tf.clip_by_value_3/clip_by_value/Minimum
*model_2/tf.clip_by_value_3/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*model_2/tf.clip_by_value_3/clip_by_value/yü
(model_2/tf.clip_by_value_3/clip_by_valueMaximum4model_2/tf.clip_by_value_3/clip_by_value/Minimum:z:03model_2/tf.clip_by_value_3/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(model_2/tf.clip_by_value_3/clip_by_value
+model_2/tf.compat.v1.floor_div_2/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2-
+model_2/tf.compat.v1.floor_div_2/FloorDiv/yø
)model_2/tf.compat.v1.floor_div_2/FloorDivFloorDiv,model_2/tf.clip_by_value_3/clip_by_value:z:04model_2/tf.compat.v1.floor_div_2/FloorDiv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)model_2/tf.compat.v1.floor_div_2/FloorDivò
,model_2/tf.math.greater_equal_3/GreaterEqualGreaterEqual"model_2/flatten_2/Reshape:output:0.model_2_tf_math_greater_equal_3_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,model_2/tf.math.greater_equal_3/GreaterEqual
%model_2/tf.math.floormod_2/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2'
%model_2/tf.math.floormod_2/FloorMod/yæ
#model_2/tf.math.floormod_2/FloorModFloorMod,model_2/tf.clip_by_value_3/clip_by_value:z:0.model_2/tf.math.floormod_2/FloorMod/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#model_2/tf.math.floormod_2/FloorMod«
model_2/embedding_8/CastCast,model_2/tf.clip_by_value_3/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_2/embedding_8/CastÞ
$model_2/embedding_8/embedding_lookupResourceGather)model_2_embedding_8_embedding_lookup_5600model_2/embedding_8/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*<
_class2
0.loc:@model_2/embedding_8/embedding_lookup/5600*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$model_2/embedding_8/embedding_lookup¼
-model_2/embedding_8/embedding_lookup/IdentityIdentity-model_2/embedding_8/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*<
_class2
0.loc:@model_2/embedding_8/embedding_lookup/5600*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-model_2/embedding_8/embedding_lookup/IdentityÝ
/model_2/embedding_8/embedding_lookup/Identity_1Identity6model_2/embedding_8/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/model_2/embedding_8/embedding_lookup/Identity_1¬
model_2/embedding_6/CastCast-model_2/tf.compat.v1.floor_div_2/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_2/embedding_6/CastÞ
$model_2/embedding_6/embedding_lookupResourceGather)model_2_embedding_6_embedding_lookup_5606model_2/embedding_6/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*<
_class2
0.loc:@model_2/embedding_6/embedding_lookup/5606*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$model_2/embedding_6/embedding_lookup¼
-model_2/embedding_6/embedding_lookup/IdentityIdentity-model_2/embedding_6/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*<
_class2
0.loc:@model_2/embedding_6/embedding_lookup/5606*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-model_2/embedding_6/embedding_lookup/IdentityÝ
/model_2/embedding_6/embedding_lookup/Identity_1Identity6model_2/embedding_6/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/model_2/embedding_6/embedding_lookup/Identity_1«
model_2/tf.cast_3/CastCast0model_2/tf.math.greater_equal_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_2/tf.cast_3/Cast
$model_2/tf.__operators__.add_6/AddV2AddV28model_2/embedding_8/embedding_lookup/Identity_1:output:08model_2/embedding_6/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$model_2/tf.__operators__.add_6/AddV2¦
model_2/embedding_7/CastCast'model_2/tf.math.floormod_2/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_2/embedding_7/CastÞ
$model_2/embedding_7/embedding_lookupResourceGather)model_2_embedding_7_embedding_lookup_5614model_2/embedding_7/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*<
_class2
0.loc:@model_2/embedding_7/embedding_lookup/5614*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$model_2/embedding_7/embedding_lookup¼
-model_2/embedding_7/embedding_lookup/IdentityIdentity-model_2/embedding_7/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*<
_class2
0.loc:@model_2/embedding_7/embedding_lookup/5614*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-model_2/embedding_7/embedding_lookup/IdentityÝ
/model_2/embedding_7/embedding_lookup/Identity_1Identity6model_2/embedding_7/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/model_2/embedding_7/embedding_lookup/Identity_1ð
$model_2/tf.__operators__.add_7/AddV2AddV2(model_2/tf.__operators__.add_6/AddV2:z:08model_2/embedding_7/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$model_2/tf.__operators__.add_7/AddV2
'model_2/tf.expand_dims_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2)
'model_2/tf.expand_dims_2/ExpandDims/dimÜ
#model_2/tf.expand_dims_2/ExpandDims
ExpandDimsmodel_2/tf.cast_3/Cast:y:00model_2/tf.expand_dims_2/ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#model_2/tf.expand_dims_2/ExpandDimsÖ
model_2/tf.math.multiply_2/MulMul(model_2/tf.__operators__.add_7/AddV2:z:0,model_2/tf.expand_dims_2/ExpandDims:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
model_2/tf.math.multiply_2/Mulª
2model_2/tf.math.reduce_sum_2/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :24
2model_2/tf.math.reduce_sum_2/Sum/reduction_indicesß
 model_2/tf.math.reduce_sum_2/SumSum"model_2/tf.math.multiply_2/Mul:z:0;model_2/tf.math.reduce_sum_2/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 model_2/tf.math.reduce_sum_2/Sum
model_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
model_3/flatten_3/Const¡
model_3/flatten_3/ReshapeReshape
inputs_0_1 model_3/flatten_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_3/flatten_3/Reshape­
2model_3/tf.clip_by_value_4/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI24
2model_3/tf.clip_by_value_4/clip_by_value/Minimum/y
0model_3/tf.clip_by_value_4/clip_by_value/MinimumMinimum"model_3/flatten_3/Reshape:output:0;model_3/tf.clip_by_value_4/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0model_3/tf.clip_by_value_4/clip_by_value/Minimum
*model_3/tf.clip_by_value_4/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*model_3/tf.clip_by_value_4/clip_by_value/yü
(model_3/tf.clip_by_value_4/clip_by_valueMaximum4model_3/tf.clip_by_value_4/clip_by_value/Minimum:z:03model_3/tf.clip_by_value_4/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(model_3/tf.clip_by_value_4/clip_by_value
+model_3/tf.compat.v1.floor_div_3/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2-
+model_3/tf.compat.v1.floor_div_3/FloorDiv/yø
)model_3/tf.compat.v1.floor_div_3/FloorDivFloorDiv,model_3/tf.clip_by_value_4/clip_by_value:z:04model_3/tf.compat.v1.floor_div_3/FloorDiv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)model_3/tf.compat.v1.floor_div_3/FloorDivò
,model_3/tf.math.greater_equal_4/GreaterEqualGreaterEqual"model_3/flatten_3/Reshape:output:0.model_3_tf_math_greater_equal_4_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,model_3/tf.math.greater_equal_4/GreaterEqual
%model_3/tf.math.floormod_3/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2'
%model_3/tf.math.floormod_3/FloorMod/yæ
#model_3/tf.math.floormod_3/FloorModFloorMod,model_3/tf.clip_by_value_4/clip_by_value:z:0.model_3/tf.math.floormod_3/FloorMod/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#model_3/tf.math.floormod_3/FloorMod­
model_3/embedding_11/CastCast,model_3/tf.clip_by_value_4/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_3/embedding_11/Castã
%model_3/embedding_11/embedding_lookupResourceGather*model_3_embedding_11_embedding_lookup_5638model_3/embedding_11/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*=
_class3
1/loc:@model_3/embedding_11/embedding_lookup/5638*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02'
%model_3/embedding_11/embedding_lookupÀ
.model_3/embedding_11/embedding_lookup/IdentityIdentity.model_3/embedding_11/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*=
_class3
1/loc:@model_3/embedding_11/embedding_lookup/5638*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.model_3/embedding_11/embedding_lookup/Identityà
0model_3/embedding_11/embedding_lookup/Identity_1Identity7model_3/embedding_11/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0model_3/embedding_11/embedding_lookup/Identity_1¬
model_3/embedding_9/CastCast-model_3/tf.compat.v1.floor_div_3/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_3/embedding_9/CastÞ
$model_3/embedding_9/embedding_lookupResourceGather)model_3_embedding_9_embedding_lookup_5644model_3/embedding_9/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*<
_class2
0.loc:@model_3/embedding_9/embedding_lookup/5644*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$model_3/embedding_9/embedding_lookup¼
-model_3/embedding_9/embedding_lookup/IdentityIdentity-model_3/embedding_9/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*<
_class2
0.loc:@model_3/embedding_9/embedding_lookup/5644*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-model_3/embedding_9/embedding_lookup/IdentityÝ
/model_3/embedding_9/embedding_lookup/Identity_1Identity6model_3/embedding_9/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/model_3/embedding_9/embedding_lookup/Identity_1«
model_3/tf.cast_4/CastCast0model_3/tf.math.greater_equal_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_3/tf.cast_4/Cast
$model_3/tf.__operators__.add_8/AddV2AddV29model_3/embedding_11/embedding_lookup/Identity_1:output:08model_3/embedding_9/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$model_3/tf.__operators__.add_8/AddV2¨
model_3/embedding_10/CastCast'model_3/tf.math.floormod_3/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_3/embedding_10/Castã
%model_3/embedding_10/embedding_lookupResourceGather*model_3_embedding_10_embedding_lookup_5652model_3/embedding_10/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*=
_class3
1/loc:@model_3/embedding_10/embedding_lookup/5652*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02'
%model_3/embedding_10/embedding_lookupÀ
.model_3/embedding_10/embedding_lookup/IdentityIdentity.model_3/embedding_10/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*=
_class3
1/loc:@model_3/embedding_10/embedding_lookup/5652*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.model_3/embedding_10/embedding_lookup/Identityà
0model_3/embedding_10/embedding_lookup/Identity_1Identity7model_3/embedding_10/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0model_3/embedding_10/embedding_lookup/Identity_1ñ
$model_3/tf.__operators__.add_9/AddV2AddV2(model_3/tf.__operators__.add_8/AddV2:z:09model_3/embedding_10/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$model_3/tf.__operators__.add_9/AddV2
'model_3/tf.expand_dims_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2)
'model_3/tf.expand_dims_3/ExpandDims/dimÜ
#model_3/tf.expand_dims_3/ExpandDims
ExpandDimsmodel_3/tf.cast_4/Cast:y:00model_3/tf.expand_dims_3/ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#model_3/tf.expand_dims_3/ExpandDimsÖ
model_3/tf.math.multiply_3/MulMul(model_3/tf.__operators__.add_9/AddV2:z:0,model_3/tf.expand_dims_3/ExpandDims:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
model_3/tf.math.multiply_3/Mulª
2model_3/tf.math.reduce_sum_3/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :24
2model_3/tf.math.reduce_sum_3/Sum/reduction_indicesß
 model_3/tf.math.reduce_sum_3/SumSum"model_3/tf.math.multiply_3/Mul:z:0;model_3/tf.math.reduce_sum_3/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 model_3/tf.math.reduce_sum_3/SumÇ
(tf.clip_by_value_5/clip_by_value/MinimumMinimuminputs_1*tf_clip_by_value_5_clip_by_value_minimum_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2*
(tf.clip_by_value_5/clip_by_value/MinimumÓ
 tf.clip_by_value_5/clip_by_valueMaximum,tf.clip_by_value_5/clip_by_value/Minimum:z:0"tf_clip_by_value_5_clip_by_value_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2"
 tf.clip_by_value_5/clip_by_value
tf.cast_5/CastCast(tf.math.greater_equal_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
tf.cast_5/Castt
tf.concat_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_3/concat/axisè
tf.concat_3/concatConcatV2)model_2/tf.math.reduce_sum_2/Sum:output:0)model_3/tf.math.reduce_sum_3/Sum:output:0 tf.concat_3/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.concat_3/concat}
tf.concat_4/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
tf.concat_4/concat/axisË
tf.concat_4/concatConcatV2$tf.clip_by_value_5/clip_by_value:z:0tf.cast_5/Cast:y:0 tf.concat_4/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.concat_4/concat§
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_9/MatMul/ReadVariableOp¡
dense_9/MatMulMatMultf.concat_3/concat:output:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_9/MatMul¥
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_9/BiasAdd/ReadVariableOp¢
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_9/BiasAddq
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_9/Relu©
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02 
dense_12/MatMul/ReadVariableOp¤
dense_12/MatMulMatMultf.concat_4/concat:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_12/MatMul¨
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_12/BiasAdd/ReadVariableOp¦
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_12/BiasAddª
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_10/MatMul/ReadVariableOp£
dense_10/MatMulMatMuldense_9/Relu:activations:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_10/MatMul¨
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_10/BiasAdd/ReadVariableOp¦
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_10/BiasAddt
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_10/Reluª
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_11/MatMul/ReadVariableOp¤
dense_11/MatMulMatMuldense_10/Relu:activations:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_11/MatMul¨
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_11/BiasAdd/ReadVariableOp¦
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_11/BiasAddt
dense_11/ReluReludense_11/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_11/Reluª
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_13/MatMul/ReadVariableOp¢
dense_13/MatMulMatMuldense_12/BiasAdd:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_13/MatMul¨
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_13/BiasAdd/ReadVariableOp¦
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_13/BiasAdd}
tf.concat_5/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
tf.concat_5/concat/axisÊ
tf.concat_5/concatConcatV2dense_11/Relu:activations:0dense_13/BiasAdd:output:0 tf.concat_5/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.concat_5/concatª
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_14/MatMul/ReadVariableOp¤
dense_14/MatMulMatMultf.concat_5/concat:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_14/MatMul¨
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_14/BiasAdd/ReadVariableOp¦
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_14/BiasAdd|
tf.nn.relu_3/ReluReludense_14/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.nn.relu_3/Reluª
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_15/MatMul/ReadVariableOp¨
dense_15/MatMulMatMultf.nn.relu_3/Relu:activations:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_15/MatMul¨
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_15/BiasAdd/ReadVariableOp¦
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_15/BiasAdd¶
tf.__operators__.add_10/AddV2AddV2dense_15/BiasAdd:output:0tf.nn.relu_3/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_10/AddV2
tf.nn.relu_4/ReluRelu!tf.__operators__.add_10/AddV2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.nn.relu_4/Reluª
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_16/MatMul/ReadVariableOp¨
dense_16/MatMulMatMultf.nn.relu_4/Relu:activations:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_16/MatMul¨
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_16/BiasAdd/ReadVariableOp¦
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_16/BiasAdd¶
tf.__operators__.add_11/AddV2AddV2dense_16/BiasAdd:output:0tf.nn.relu_4/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_11/AddV2
tf.nn.relu_5/ReluRelu!tf.__operators__.add_11/AddV2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.nn.relu_5/Reluá
2normalize_1/normalization_1/Reshape/ReadVariableOpReadVariableOp;normalize_1_normalization_1_reshape_readvariableop_resource*
_output_shapes	
:*
dtype024
2normalize_1/normalization_1/Reshape/ReadVariableOp§
)normalize_1/normalization_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2+
)normalize_1/normalization_1/Reshape/shapeï
#normalize_1/normalization_1/ReshapeReshape:normalize_1/normalization_1/Reshape/ReadVariableOp:value:02normalize_1/normalization_1/Reshape/shape:output:0*
T0*
_output_shapes
:	2%
#normalize_1/normalization_1/Reshapeç
4normalize_1/normalization_1/Reshape_1/ReadVariableOpReadVariableOp=normalize_1_normalization_1_reshape_1_readvariableop_resource*
_output_shapes	
:*
dtype026
4normalize_1/normalization_1/Reshape_1/ReadVariableOp«
+normalize_1/normalization_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2-
+normalize_1/normalization_1/Reshape_1/shape÷
%normalize_1/normalization_1/Reshape_1Reshape<normalize_1/normalization_1/Reshape_1/ReadVariableOp:value:04normalize_1/normalization_1/Reshape_1/shape:output:0*
T0*
_output_shapes
:	2'
%normalize_1/normalization_1/Reshape_1Ë
normalize_1/normalization_1/subSubtf.nn.relu_5/Relu:activations:0,normalize_1/normalization_1/Reshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
normalize_1/normalization_1/sub¦
 normalize_1/normalization_1/SqrtSqrt.normalize_1/normalization_1/Reshape_1:output:0*
T0*
_output_shapes
:	2"
 normalize_1/normalization_1/Sqrt
%normalize_1/normalization_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö32'
%normalize_1/normalization_1/Maximum/yÕ
#normalize_1/normalization_1/MaximumMaximum$normalize_1/normalization_1/Sqrt:y:0.normalize_1/normalization_1/Maximum/y:output:0*
T0*
_output_shapes
:	2%
#normalize_1/normalization_1/MaximumÖ
#normalize_1/normalization_1/truedivRealDiv#normalize_1/normalization_1/sub:z:0'normalize_1/normalization_1/Maximum:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#normalize_1/normalization_1/truediv©
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02 
dense_17/MatMul/ReadVariableOp¯
dense_17/MatMulMatMul'normalize_1/normalization_1/truediv:z:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_17/MatMul§
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_17/BiasAdd/ReadVariableOp¥
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_17/BiasAdd
IdentityIdentitydense_17/BiasAdd:output:0 ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp%^model_2/embedding_6/embedding_lookup%^model_2/embedding_7/embedding_lookup%^model_2/embedding_8/embedding_lookup&^model_3/embedding_10/embedding_lookup&^model_3/embedding_11/embedding_lookup%^model_3/embedding_9/embedding_lookup3^normalize_1/normalization_1/Reshape/ReadVariableOp5^normalize_1/normalization_1/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*À
_input_shapes®
«:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
: : :::: :::: : ::::::::::::::::::::2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2L
$model_2/embedding_6/embedding_lookup$model_2/embedding_6/embedding_lookup2L
$model_2/embedding_7/embedding_lookup$model_2/embedding_7/embedding_lookup2L
$model_2/embedding_8/embedding_lookup$model_2/embedding_8/embedding_lookup2N
%model_3/embedding_10/embedding_lookup%model_3/embedding_10/embedding_lookup2N
%model_3/embedding_11/embedding_lookup%model_3/embedding_11/embedding_lookup2L
$model_3/embedding_9/embedding_lookup$model_3/embedding_9/embedding_lookup2h
2normalize_1/normalization_1/Reshape/ReadVariableOp2normalize_1/normalization_1/Reshape/ReadVariableOp2l
4normalize_1/normalization_1/Reshape_1/ReadVariableOp4normalize_1/normalization_1/Reshape_1/ReadVariableOp:S O
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/0/0:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/0/1:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
¹

&__inference_model_3_layer_call_fn_6278

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_model_3_layer_call_and_return_conditional_losses_47182
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 
Ç

E__inference_normalize_1_layer_call_and_return_conditional_losses_6450
x3
/normalization_1_reshape_readvariableop_resource5
1normalization_1_reshape_1_readvariableop_resource
identity¢&normalization_1/Reshape/ReadVariableOp¢(normalization_1/Reshape_1/ReadVariableOp½
&normalization_1/Reshape/ReadVariableOpReadVariableOp/normalization_1_reshape_readvariableop_resource*
_output_shapes	
:*
dtype02(
&normalization_1/Reshape/ReadVariableOp
normalization_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_1/Reshape/shape¿
normalization_1/ReshapeReshape.normalization_1/Reshape/ReadVariableOp:value:0&normalization_1/Reshape/shape:output:0*
T0*
_output_shapes
:	2
normalization_1/ReshapeÃ
(normalization_1/Reshape_1/ReadVariableOpReadVariableOp1normalization_1_reshape_1_readvariableop_resource*
_output_shapes	
:*
dtype02*
(normalization_1/Reshape_1/ReadVariableOp
normalization_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_1/Reshape_1/shapeÇ
normalization_1/Reshape_1Reshape0normalization_1/Reshape_1/ReadVariableOp:value:0(normalization_1/Reshape_1/shape:output:0*
T0*
_output_shapes
:	2
normalization_1/Reshape_1
normalization_1/subSubx normalization_1/Reshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
normalization_1/sub
normalization_1/SqrtSqrt"normalization_1/Reshape_1:output:0*
T0*
_output_shapes
:	2
normalization_1/Sqrt{
normalization_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö32
normalization_1/Maximum/y¥
normalization_1/MaximumMaximumnormalization_1/Sqrt:y:0"normalization_1/Maximum/y:output:0*
T0*
_output_shapes
:	2
normalization_1/Maximum¦
normalization_1/truedivRealDivnormalization_1/sub:z:0normalization_1/Maximum:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
normalization_1/truedivÄ
IdentityIdentitynormalization_1/truediv:z:0'^normalization_1/Reshape/ReadVariableOp)^normalization_1/Reshape_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::2P
&normalization_1/Reshape/ReadVariableOp&normalization_1/Reshape/ReadVariableOp2T
(normalization_1/Reshape_1/ReadVariableOp(normalization_1/Reshape_1/ReadVariableOp:K G
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
¼

&__inference_model_2_layer_call_fn_4503
input_3
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_model_2_layer_call_and_return_conditional_losses_44922
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: :::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_3:

_output_shapes
: 
Öx

 __inference__traced_restore_6816
file_prefix#
assignvariableop_dense_9_kernel#
assignvariableop_1_dense_9_bias&
"assignvariableop_2_dense_10_kernel$
 assignvariableop_3_dense_10_bias&
"assignvariableop_4_dense_12_kernel$
 assignvariableop_5_dense_12_bias&
"assignvariableop_6_dense_11_kernel$
 assignvariableop_7_dense_11_bias&
"assignvariableop_8_dense_13_kernel$
 assignvariableop_9_dense_13_bias'
#assignvariableop_10_dense_14_kernel%
!assignvariableop_11_dense_14_bias'
#assignvariableop_12_dense_15_kernel%
!assignvariableop_13_dense_15_bias'
#assignvariableop_14_dense_16_kernel%
!assignvariableop_15_dense_16_bias'
#assignvariableop_16_dense_17_kernel%
!assignvariableop_17_dense_17_bias.
*assignvariableop_18_embedding_8_embeddings.
*assignvariableop_19_embedding_6_embeddings.
*assignvariableop_20_embedding_7_embeddings/
+assignvariableop_21_embedding_11_embeddings.
*assignvariableop_22_embedding_9_embeddings/
+assignvariableop_23_embedding_10_embeddings8
4assignvariableop_24_normalize_1_normalization_1_mean<
8assignvariableop_25_normalize_1_normalization_1_variance9
5assignvariableop_26_normalize_1_normalization_1_count
assignvariableop_27_total
assignvariableop_28_count
identity_30¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*¦
valueBB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesÊ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesÂ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesz
x::::::::::::::::::::::::::::::*,
dtypes"
 2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_dense_9_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¤
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_9_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2§
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_10_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¥
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_10_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4§
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_12_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¥
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_12_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6§
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_11_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¥
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_11_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8§
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_13_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¥
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_13_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10«
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_14_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11©
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_14_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12«
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_15_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13©
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_15_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14«
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_16_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15©
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_16_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16«
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_17_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17©
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense_17_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18²
AssignVariableOp_18AssignVariableOp*assignvariableop_18_embedding_8_embeddingsIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19²
AssignVariableOp_19AssignVariableOp*assignvariableop_19_embedding_6_embeddingsIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20²
AssignVariableOp_20AssignVariableOp*assignvariableop_20_embedding_7_embeddingsIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21³
AssignVariableOp_21AssignVariableOp+assignvariableop_21_embedding_11_embeddingsIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22²
AssignVariableOp_22AssignVariableOp*assignvariableop_22_embedding_9_embeddingsIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23³
AssignVariableOp_23AssignVariableOp+assignvariableop_23_embedding_10_embeddingsIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24¼
AssignVariableOp_24AssignVariableOp4assignvariableop_24_normalize_1_normalization_1_meanIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25À
AssignVariableOp_25AssignVariableOp8assignvariableop_25_normalize_1_normalization_1_varianceIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_26½
AssignVariableOp_26AssignVariableOp5assignvariableop_26_normalize_1_normalization_1_countIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27¡
AssignVariableOp_27AssignVariableOpassignvariableop_27_totalIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28¡
AssignVariableOp_28AssignVariableOpassignvariableop_28_countIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_289
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpÜ
Identity_29Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_29Ï
Identity_30IdentityIdentity_29:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_30"#
identity_30Identity_30:output:0*
_input_shapesx
v: :::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ù	

E__inference_embedding_6_layer_call_and_return_conditional_losses_4337

inputs
embedding_lookup_4331
identity¢embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Castú
embedding_lookupResourceGatherembedding_lookup_4331Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*(
_class
loc:@embedding_lookup/4331*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_lookupì
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@embedding_lookup/4331*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity¡
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨
ß
-__inference_custom_model_1_layer_call_fn_5348

cards0

cards1
bets
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallcards0cards1betsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
 !*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_custom_model_1_layer_call_and_return_conditional_losses_52832
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*À
_input_shapes®
«:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
: : :::: :::: : ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namecards0:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namecards1:MI
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


_user_specified_namebets:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
	
Û
B__inference_dense_12_layer_call_and_return_conditional_losses_4853

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì
ë
-__inference_custom_model_1_layer_call_fn_5989

inputs_0_0

inputs_0_1
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29
identity¢StatefulPartitionedCall¡
StatefulPartitionedCallStatefulPartitionedCall
inputs_0_0
inputs_0_1inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
 !*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_custom_model_1_layer_call_and_return_conditional_losses_52832
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*À
_input_shapes®
«:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
: : :::: :::: : ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/0/0:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/0/1:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
	
Û
B__inference_dense_16_layer_call_and_return_conditional_losses_5016

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È¸

__inference__wrapped_model_4277

cards0

cards1
bets9
5custom_model_1_tf_math_greater_equal_5_greaterequal_yA
=custom_model_1_model_2_tf_math_greater_equal_3_greaterequal_y<
8custom_model_1_model_2_embedding_8_embedding_lookup_4127<
8custom_model_1_model_2_embedding_6_embedding_lookup_4133<
8custom_model_1_model_2_embedding_7_embedding_lookup_4141A
=custom_model_1_model_3_tf_math_greater_equal_4_greaterequal_y=
9custom_model_1_model_3_embedding_11_embedding_lookup_4165<
8custom_model_1_model_3_embedding_9_embedding_lookup_4171=
9custom_model_1_model_3_embedding_10_embedding_lookup_4179=
9custom_model_1_tf_clip_by_value_5_clip_by_value_minimum_y5
1custom_model_1_tf_clip_by_value_5_clip_by_value_y9
5custom_model_1_dense_9_matmul_readvariableop_resource:
6custom_model_1_dense_9_biasadd_readvariableop_resource:
6custom_model_1_dense_12_matmul_readvariableop_resource;
7custom_model_1_dense_12_biasadd_readvariableop_resource:
6custom_model_1_dense_10_matmul_readvariableop_resource;
7custom_model_1_dense_10_biasadd_readvariableop_resource:
6custom_model_1_dense_11_matmul_readvariableop_resource;
7custom_model_1_dense_11_biasadd_readvariableop_resource:
6custom_model_1_dense_13_matmul_readvariableop_resource;
7custom_model_1_dense_13_biasadd_readvariableop_resource:
6custom_model_1_dense_14_matmul_readvariableop_resource;
7custom_model_1_dense_14_biasadd_readvariableop_resource:
6custom_model_1_dense_15_matmul_readvariableop_resource;
7custom_model_1_dense_15_biasadd_readvariableop_resource:
6custom_model_1_dense_16_matmul_readvariableop_resource;
7custom_model_1_dense_16_biasadd_readvariableop_resourceN
Jcustom_model_1_normalize_1_normalization_1_reshape_readvariableop_resourceP
Lcustom_model_1_normalize_1_normalization_1_reshape_1_readvariableop_resource:
6custom_model_1_dense_17_matmul_readvariableop_resource;
7custom_model_1_dense_17_biasadd_readvariableop_resource
identity¢.custom_model_1/dense_10/BiasAdd/ReadVariableOp¢-custom_model_1/dense_10/MatMul/ReadVariableOp¢.custom_model_1/dense_11/BiasAdd/ReadVariableOp¢-custom_model_1/dense_11/MatMul/ReadVariableOp¢.custom_model_1/dense_12/BiasAdd/ReadVariableOp¢-custom_model_1/dense_12/MatMul/ReadVariableOp¢.custom_model_1/dense_13/BiasAdd/ReadVariableOp¢-custom_model_1/dense_13/MatMul/ReadVariableOp¢.custom_model_1/dense_14/BiasAdd/ReadVariableOp¢-custom_model_1/dense_14/MatMul/ReadVariableOp¢.custom_model_1/dense_15/BiasAdd/ReadVariableOp¢-custom_model_1/dense_15/MatMul/ReadVariableOp¢.custom_model_1/dense_16/BiasAdd/ReadVariableOp¢-custom_model_1/dense_16/MatMul/ReadVariableOp¢.custom_model_1/dense_17/BiasAdd/ReadVariableOp¢-custom_model_1/dense_17/MatMul/ReadVariableOp¢-custom_model_1/dense_9/BiasAdd/ReadVariableOp¢,custom_model_1/dense_9/MatMul/ReadVariableOp¢3custom_model_1/model_2/embedding_6/embedding_lookup¢3custom_model_1/model_2/embedding_7/embedding_lookup¢3custom_model_1/model_2/embedding_8/embedding_lookup¢4custom_model_1/model_3/embedding_10/embedding_lookup¢4custom_model_1/model_3/embedding_11/embedding_lookup¢3custom_model_1/model_3/embedding_9/embedding_lookup¢Acustom_model_1/normalize_1/normalization_1/Reshape/ReadVariableOp¢Ccustom_model_1/normalize_1/normalization_1/Reshape_1/ReadVariableOpé
3custom_model_1/tf.math.greater_equal_5/GreaterEqualGreaterEqualbets5custom_model_1_tf_math_greater_equal_5_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
25
3custom_model_1/tf.math.greater_equal_5/GreaterEqual¡
&custom_model_1/model_2/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2(
&custom_model_1/model_2/flatten_2/ConstÊ
(custom_model_1/model_2/flatten_2/ReshapeReshapecards0/custom_model_1/model_2/flatten_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(custom_model_1/model_2/flatten_2/ReshapeË
Acustom_model_1/model_2/tf.clip_by_value_3/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2C
Acustom_model_1/model_2/tf.clip_by_value_3/clip_by_value/Minimum/y¾
?custom_model_1/model_2/tf.clip_by_value_3/clip_by_value/MinimumMinimum1custom_model_1/model_2/flatten_2/Reshape:output:0Jcustom_model_1/model_2/tf.clip_by_value_3/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2A
?custom_model_1/model_2/tf.clip_by_value_3/clip_by_value/Minimum»
9custom_model_1/model_2/tf.clip_by_value_3/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2;
9custom_model_1/model_2/tf.clip_by_value_3/clip_by_value/y¸
7custom_model_1/model_2/tf.clip_by_value_3/clip_by_valueMaximumCcustom_model_1/model_2/tf.clip_by_value_3/clip_by_value/Minimum:z:0Bcustom_model_1/model_2/tf.clip_by_value_3/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ29
7custom_model_1/model_2/tf.clip_by_value_3/clip_by_value½
:custom_model_1/model_2/tf.compat.v1.floor_div_2/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2<
:custom_model_1/model_2/tf.compat.v1.floor_div_2/FloorDiv/y´
8custom_model_1/model_2/tf.compat.v1.floor_div_2/FloorDivFloorDiv;custom_model_1/model_2/tf.clip_by_value_3/clip_by_value:z:0Ccustom_model_1/model_2/tf.compat.v1.floor_div_2/FloorDiv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:
8custom_model_1/model_2/tf.compat.v1.floor_div_2/FloorDiv®
;custom_model_1/model_2/tf.math.greater_equal_3/GreaterEqualGreaterEqual1custom_model_1/model_2/flatten_2/Reshape:output:0=custom_model_1_model_2_tf_math_greater_equal_3_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2=
;custom_model_1/model_2/tf.math.greater_equal_3/GreaterEqual±
4custom_model_1/model_2/tf.math.floormod_2/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @26
4custom_model_1/model_2/tf.math.floormod_2/FloorMod/y¢
2custom_model_1/model_2/tf.math.floormod_2/FloorModFloorMod;custom_model_1/model_2/tf.clip_by_value_3/clip_by_value:z:0=custom_model_1/model_2/tf.math.floormod_2/FloorMod/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2custom_model_1/model_2/tf.math.floormod_2/FloorModØ
'custom_model_1/model_2/embedding_8/CastCast;custom_model_1/model_2/tf.clip_by_value_3/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'custom_model_1/model_2/embedding_8/Cast©
3custom_model_1/model_2/embedding_8/embedding_lookupResourceGather8custom_model_1_model_2_embedding_8_embedding_lookup_4127+custom_model_1/model_2/embedding_8/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*K
_classA
?=loc:@custom_model_1/model_2/embedding_8/embedding_lookup/4127*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype025
3custom_model_1/model_2/embedding_8/embedding_lookupø
<custom_model_1/model_2/embedding_8/embedding_lookup/IdentityIdentity<custom_model_1/model_2/embedding_8/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*K
_classA
?=loc:@custom_model_1/model_2/embedding_8/embedding_lookup/4127*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2>
<custom_model_1/model_2/embedding_8/embedding_lookup/Identity
>custom_model_1/model_2/embedding_8/embedding_lookup/Identity_1IdentityEcustom_model_1/model_2/embedding_8/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2@
>custom_model_1/model_2/embedding_8/embedding_lookup/Identity_1Ù
'custom_model_1/model_2/embedding_6/CastCast<custom_model_1/model_2/tf.compat.v1.floor_div_2/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'custom_model_1/model_2/embedding_6/Cast©
3custom_model_1/model_2/embedding_6/embedding_lookupResourceGather8custom_model_1_model_2_embedding_6_embedding_lookup_4133+custom_model_1/model_2/embedding_6/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*K
_classA
?=loc:@custom_model_1/model_2/embedding_6/embedding_lookup/4133*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype025
3custom_model_1/model_2/embedding_6/embedding_lookupø
<custom_model_1/model_2/embedding_6/embedding_lookup/IdentityIdentity<custom_model_1/model_2/embedding_6/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*K
_classA
?=loc:@custom_model_1/model_2/embedding_6/embedding_lookup/4133*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2>
<custom_model_1/model_2/embedding_6/embedding_lookup/Identity
>custom_model_1/model_2/embedding_6/embedding_lookup/Identity_1IdentityEcustom_model_1/model_2/embedding_6/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2@
>custom_model_1/model_2/embedding_6/embedding_lookup/Identity_1Ø
%custom_model_1/model_2/tf.cast_3/CastCast?custom_model_1/model_2/tf.math.greater_equal_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%custom_model_1/model_2/tf.cast_3/Cast¼
3custom_model_1/model_2/tf.__operators__.add_6/AddV2AddV2Gcustom_model_1/model_2/embedding_8/embedding_lookup/Identity_1:output:0Gcustom_model_1/model_2/embedding_6/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3custom_model_1/model_2/tf.__operators__.add_6/AddV2Ó
'custom_model_1/model_2/embedding_7/CastCast6custom_model_1/model_2/tf.math.floormod_2/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'custom_model_1/model_2/embedding_7/Cast©
3custom_model_1/model_2/embedding_7/embedding_lookupResourceGather8custom_model_1_model_2_embedding_7_embedding_lookup_4141+custom_model_1/model_2/embedding_7/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*K
_classA
?=loc:@custom_model_1/model_2/embedding_7/embedding_lookup/4141*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype025
3custom_model_1/model_2/embedding_7/embedding_lookupø
<custom_model_1/model_2/embedding_7/embedding_lookup/IdentityIdentity<custom_model_1/model_2/embedding_7/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*K
_classA
?=loc:@custom_model_1/model_2/embedding_7/embedding_lookup/4141*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2>
<custom_model_1/model_2/embedding_7/embedding_lookup/Identity
>custom_model_1/model_2/embedding_7/embedding_lookup/Identity_1IdentityEcustom_model_1/model_2/embedding_7/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2@
>custom_model_1/model_2/embedding_7/embedding_lookup/Identity_1¬
3custom_model_1/model_2/tf.__operators__.add_7/AddV2AddV27custom_model_1/model_2/tf.__operators__.add_6/AddV2:z:0Gcustom_model_1/model_2/embedding_7/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3custom_model_1/model_2/tf.__operators__.add_7/AddV2»
6custom_model_1/model_2/tf.expand_dims_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ28
6custom_model_1/model_2/tf.expand_dims_2/ExpandDims/dim
2custom_model_1/model_2/tf.expand_dims_2/ExpandDims
ExpandDims)custom_model_1/model_2/tf.cast_3/Cast:y:0?custom_model_1/model_2/tf.expand_dims_2/ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2custom_model_1/model_2/tf.expand_dims_2/ExpandDims
-custom_model_1/model_2/tf.math.multiply_2/MulMul7custom_model_1/model_2/tf.__operators__.add_7/AddV2:z:0;custom_model_1/model_2/tf.expand_dims_2/ExpandDims:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-custom_model_1/model_2/tf.math.multiply_2/MulÈ
Acustom_model_1/model_2/tf.math.reduce_sum_2/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2C
Acustom_model_1/model_2/tf.math.reduce_sum_2/Sum/reduction_indices
/custom_model_1/model_2/tf.math.reduce_sum_2/SumSum1custom_model_1/model_2/tf.math.multiply_2/Mul:z:0Jcustom_model_1/model_2/tf.math.reduce_sum_2/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/custom_model_1/model_2/tf.math.reduce_sum_2/Sum¡
&custom_model_1/model_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2(
&custom_model_1/model_3/flatten_3/ConstÊ
(custom_model_1/model_3/flatten_3/ReshapeReshapecards1/custom_model_1/model_3/flatten_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(custom_model_1/model_3/flatten_3/ReshapeË
Acustom_model_1/model_3/tf.clip_by_value_4/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2C
Acustom_model_1/model_3/tf.clip_by_value_4/clip_by_value/Minimum/y¾
?custom_model_1/model_3/tf.clip_by_value_4/clip_by_value/MinimumMinimum1custom_model_1/model_3/flatten_3/Reshape:output:0Jcustom_model_1/model_3/tf.clip_by_value_4/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2A
?custom_model_1/model_3/tf.clip_by_value_4/clip_by_value/Minimum»
9custom_model_1/model_3/tf.clip_by_value_4/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2;
9custom_model_1/model_3/tf.clip_by_value_4/clip_by_value/y¸
7custom_model_1/model_3/tf.clip_by_value_4/clip_by_valueMaximumCcustom_model_1/model_3/tf.clip_by_value_4/clip_by_value/Minimum:z:0Bcustom_model_1/model_3/tf.clip_by_value_4/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ29
7custom_model_1/model_3/tf.clip_by_value_4/clip_by_value½
:custom_model_1/model_3/tf.compat.v1.floor_div_3/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2<
:custom_model_1/model_3/tf.compat.v1.floor_div_3/FloorDiv/y´
8custom_model_1/model_3/tf.compat.v1.floor_div_3/FloorDivFloorDiv;custom_model_1/model_3/tf.clip_by_value_4/clip_by_value:z:0Ccustom_model_1/model_3/tf.compat.v1.floor_div_3/FloorDiv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:
8custom_model_1/model_3/tf.compat.v1.floor_div_3/FloorDiv®
;custom_model_1/model_3/tf.math.greater_equal_4/GreaterEqualGreaterEqual1custom_model_1/model_3/flatten_3/Reshape:output:0=custom_model_1_model_3_tf_math_greater_equal_4_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2=
;custom_model_1/model_3/tf.math.greater_equal_4/GreaterEqual±
4custom_model_1/model_3/tf.math.floormod_3/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @26
4custom_model_1/model_3/tf.math.floormod_3/FloorMod/y¢
2custom_model_1/model_3/tf.math.floormod_3/FloorModFloorMod;custom_model_1/model_3/tf.clip_by_value_4/clip_by_value:z:0=custom_model_1/model_3/tf.math.floormod_3/FloorMod/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2custom_model_1/model_3/tf.math.floormod_3/FloorModÚ
(custom_model_1/model_3/embedding_11/CastCast;custom_model_1/model_3/tf.clip_by_value_4/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(custom_model_1/model_3/embedding_11/Cast®
4custom_model_1/model_3/embedding_11/embedding_lookupResourceGather9custom_model_1_model_3_embedding_11_embedding_lookup_4165,custom_model_1/model_3/embedding_11/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*L
_classB
@>loc:@custom_model_1/model_3/embedding_11/embedding_lookup/4165*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype026
4custom_model_1/model_3/embedding_11/embedding_lookupü
=custom_model_1/model_3/embedding_11/embedding_lookup/IdentityIdentity=custom_model_1/model_3/embedding_11/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*L
_classB
@>loc:@custom_model_1/model_3/embedding_11/embedding_lookup/4165*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=custom_model_1/model_3/embedding_11/embedding_lookup/Identity
?custom_model_1/model_3/embedding_11/embedding_lookup/Identity_1IdentityFcustom_model_1/model_3/embedding_11/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2A
?custom_model_1/model_3/embedding_11/embedding_lookup/Identity_1Ù
'custom_model_1/model_3/embedding_9/CastCast<custom_model_1/model_3/tf.compat.v1.floor_div_3/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'custom_model_1/model_3/embedding_9/Cast©
3custom_model_1/model_3/embedding_9/embedding_lookupResourceGather8custom_model_1_model_3_embedding_9_embedding_lookup_4171+custom_model_1/model_3/embedding_9/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*K
_classA
?=loc:@custom_model_1/model_3/embedding_9/embedding_lookup/4171*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype025
3custom_model_1/model_3/embedding_9/embedding_lookupø
<custom_model_1/model_3/embedding_9/embedding_lookup/IdentityIdentity<custom_model_1/model_3/embedding_9/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*K
_classA
?=loc:@custom_model_1/model_3/embedding_9/embedding_lookup/4171*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2>
<custom_model_1/model_3/embedding_9/embedding_lookup/Identity
>custom_model_1/model_3/embedding_9/embedding_lookup/Identity_1IdentityEcustom_model_1/model_3/embedding_9/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2@
>custom_model_1/model_3/embedding_9/embedding_lookup/Identity_1Ø
%custom_model_1/model_3/tf.cast_4/CastCast?custom_model_1/model_3/tf.math.greater_equal_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%custom_model_1/model_3/tf.cast_4/Cast½
3custom_model_1/model_3/tf.__operators__.add_8/AddV2AddV2Hcustom_model_1/model_3/embedding_11/embedding_lookup/Identity_1:output:0Gcustom_model_1/model_3/embedding_9/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3custom_model_1/model_3/tf.__operators__.add_8/AddV2Õ
(custom_model_1/model_3/embedding_10/CastCast6custom_model_1/model_3/tf.math.floormod_3/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(custom_model_1/model_3/embedding_10/Cast®
4custom_model_1/model_3/embedding_10/embedding_lookupResourceGather9custom_model_1_model_3_embedding_10_embedding_lookup_4179,custom_model_1/model_3/embedding_10/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*L
_classB
@>loc:@custom_model_1/model_3/embedding_10/embedding_lookup/4179*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype026
4custom_model_1/model_3/embedding_10/embedding_lookupü
=custom_model_1/model_3/embedding_10/embedding_lookup/IdentityIdentity=custom_model_1/model_3/embedding_10/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*L
_classB
@>loc:@custom_model_1/model_3/embedding_10/embedding_lookup/4179*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=custom_model_1/model_3/embedding_10/embedding_lookup/Identity
?custom_model_1/model_3/embedding_10/embedding_lookup/Identity_1IdentityFcustom_model_1/model_3/embedding_10/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2A
?custom_model_1/model_3/embedding_10/embedding_lookup/Identity_1­
3custom_model_1/model_3/tf.__operators__.add_9/AddV2AddV27custom_model_1/model_3/tf.__operators__.add_8/AddV2:z:0Hcustom_model_1/model_3/embedding_10/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3custom_model_1/model_3/tf.__operators__.add_9/AddV2»
6custom_model_1/model_3/tf.expand_dims_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ28
6custom_model_1/model_3/tf.expand_dims_3/ExpandDims/dim
2custom_model_1/model_3/tf.expand_dims_3/ExpandDims
ExpandDims)custom_model_1/model_3/tf.cast_4/Cast:y:0?custom_model_1/model_3/tf.expand_dims_3/ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2custom_model_1/model_3/tf.expand_dims_3/ExpandDims
-custom_model_1/model_3/tf.math.multiply_3/MulMul7custom_model_1/model_3/tf.__operators__.add_9/AddV2:z:0;custom_model_1/model_3/tf.expand_dims_3/ExpandDims:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-custom_model_1/model_3/tf.math.multiply_3/MulÈ
Acustom_model_1/model_3/tf.math.reduce_sum_3/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2C
Acustom_model_1/model_3/tf.math.reduce_sum_3/Sum/reduction_indices
/custom_model_1/model_3/tf.math.reduce_sum_3/SumSum1custom_model_1/model_3/tf.math.multiply_3/Mul:z:0Jcustom_model_1/model_3/tf.math.reduce_sum_3/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/custom_model_1/model_3/tf.math.reduce_sum_3/Sumð
7custom_model_1/tf.clip_by_value_5/clip_by_value/MinimumMinimumbets9custom_model_1_tf_clip_by_value_5_clip_by_value_minimum_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
29
7custom_model_1/tf.clip_by_value_5/clip_by_value/Minimum
/custom_model_1/tf.clip_by_value_5/clip_by_valueMaximum;custom_model_1/tf.clip_by_value_5/clip_by_value/Minimum:z:01custom_model_1_tf_clip_by_value_5_clip_by_value_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
21
/custom_model_1/tf.clip_by_value_5/clip_by_valueÀ
custom_model_1/tf.cast_5/CastCast7custom_model_1/tf.math.greater_equal_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
custom_model_1/tf.cast_5/Cast
&custom_model_1/tf.concat_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2(
&custom_model_1/tf.concat_3/concat/axis³
!custom_model_1/tf.concat_3/concatConcatV28custom_model_1/model_2/tf.math.reduce_sum_2/Sum:output:08custom_model_1/model_3/tf.math.reduce_sum_3/Sum:output:0/custom_model_1/tf.concat_3/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!custom_model_1/tf.concat_3/concat
&custom_model_1/tf.concat_4/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&custom_model_1/tf.concat_4/concat/axis
!custom_model_1/tf.concat_4/concatConcatV23custom_model_1/tf.clip_by_value_5/clip_by_value:z:0!custom_model_1/tf.cast_5/Cast:y:0/custom_model_1/tf.concat_4/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!custom_model_1/tf.concat_4/concatÔ
,custom_model_1/dense_9/MatMul/ReadVariableOpReadVariableOp5custom_model_1_dense_9_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02.
,custom_model_1/dense_9/MatMul/ReadVariableOpÝ
custom_model_1/dense_9/MatMulMatMul*custom_model_1/tf.concat_3/concat:output:04custom_model_1/dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
custom_model_1/dense_9/MatMulÒ
-custom_model_1/dense_9/BiasAdd/ReadVariableOpReadVariableOp6custom_model_1_dense_9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-custom_model_1/dense_9/BiasAdd/ReadVariableOpÞ
custom_model_1/dense_9/BiasAddBiasAdd'custom_model_1/dense_9/MatMul:product:05custom_model_1/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
custom_model_1/dense_9/BiasAdd
custom_model_1/dense_9/ReluRelu'custom_model_1/dense_9/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
custom_model_1/dense_9/ReluÖ
-custom_model_1/dense_12/MatMul/ReadVariableOpReadVariableOp6custom_model_1_dense_12_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02/
-custom_model_1/dense_12/MatMul/ReadVariableOpà
custom_model_1/dense_12/MatMulMatMul*custom_model_1/tf.concat_4/concat:output:05custom_model_1/dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
custom_model_1/dense_12/MatMulÕ
.custom_model_1/dense_12/BiasAdd/ReadVariableOpReadVariableOp7custom_model_1_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.custom_model_1/dense_12/BiasAdd/ReadVariableOpâ
custom_model_1/dense_12/BiasAddBiasAdd(custom_model_1/dense_12/MatMul:product:06custom_model_1/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
custom_model_1/dense_12/BiasAdd×
-custom_model_1/dense_10/MatMul/ReadVariableOpReadVariableOp6custom_model_1_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02/
-custom_model_1/dense_10/MatMul/ReadVariableOpß
custom_model_1/dense_10/MatMulMatMul)custom_model_1/dense_9/Relu:activations:05custom_model_1/dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
custom_model_1/dense_10/MatMulÕ
.custom_model_1/dense_10/BiasAdd/ReadVariableOpReadVariableOp7custom_model_1_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.custom_model_1/dense_10/BiasAdd/ReadVariableOpâ
custom_model_1/dense_10/BiasAddBiasAdd(custom_model_1/dense_10/MatMul:product:06custom_model_1/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
custom_model_1/dense_10/BiasAdd¡
custom_model_1/dense_10/ReluRelu(custom_model_1/dense_10/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
custom_model_1/dense_10/Relu×
-custom_model_1/dense_11/MatMul/ReadVariableOpReadVariableOp6custom_model_1_dense_11_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02/
-custom_model_1/dense_11/MatMul/ReadVariableOpà
custom_model_1/dense_11/MatMulMatMul*custom_model_1/dense_10/Relu:activations:05custom_model_1/dense_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
custom_model_1/dense_11/MatMulÕ
.custom_model_1/dense_11/BiasAdd/ReadVariableOpReadVariableOp7custom_model_1_dense_11_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.custom_model_1/dense_11/BiasAdd/ReadVariableOpâ
custom_model_1/dense_11/BiasAddBiasAdd(custom_model_1/dense_11/MatMul:product:06custom_model_1/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
custom_model_1/dense_11/BiasAdd¡
custom_model_1/dense_11/ReluRelu(custom_model_1/dense_11/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
custom_model_1/dense_11/Relu×
-custom_model_1/dense_13/MatMul/ReadVariableOpReadVariableOp6custom_model_1_dense_13_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02/
-custom_model_1/dense_13/MatMul/ReadVariableOpÞ
custom_model_1/dense_13/MatMulMatMul(custom_model_1/dense_12/BiasAdd:output:05custom_model_1/dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
custom_model_1/dense_13/MatMulÕ
.custom_model_1/dense_13/BiasAdd/ReadVariableOpReadVariableOp7custom_model_1_dense_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.custom_model_1/dense_13/BiasAdd/ReadVariableOpâ
custom_model_1/dense_13/BiasAddBiasAdd(custom_model_1/dense_13/MatMul:product:06custom_model_1/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
custom_model_1/dense_13/BiasAdd
&custom_model_1/tf.concat_5/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&custom_model_1/tf.concat_5/concat/axis
!custom_model_1/tf.concat_5/concatConcatV2*custom_model_1/dense_11/Relu:activations:0(custom_model_1/dense_13/BiasAdd:output:0/custom_model_1/tf.concat_5/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!custom_model_1/tf.concat_5/concat×
-custom_model_1/dense_14/MatMul/ReadVariableOpReadVariableOp6custom_model_1_dense_14_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02/
-custom_model_1/dense_14/MatMul/ReadVariableOpà
custom_model_1/dense_14/MatMulMatMul*custom_model_1/tf.concat_5/concat:output:05custom_model_1/dense_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
custom_model_1/dense_14/MatMulÕ
.custom_model_1/dense_14/BiasAdd/ReadVariableOpReadVariableOp7custom_model_1_dense_14_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.custom_model_1/dense_14/BiasAdd/ReadVariableOpâ
custom_model_1/dense_14/BiasAddBiasAdd(custom_model_1/dense_14/MatMul:product:06custom_model_1/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
custom_model_1/dense_14/BiasAdd©
 custom_model_1/tf.nn.relu_3/ReluRelu(custom_model_1/dense_14/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 custom_model_1/tf.nn.relu_3/Relu×
-custom_model_1/dense_15/MatMul/ReadVariableOpReadVariableOp6custom_model_1_dense_15_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02/
-custom_model_1/dense_15/MatMul/ReadVariableOpä
custom_model_1/dense_15/MatMulMatMul.custom_model_1/tf.nn.relu_3/Relu:activations:05custom_model_1/dense_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
custom_model_1/dense_15/MatMulÕ
.custom_model_1/dense_15/BiasAdd/ReadVariableOpReadVariableOp7custom_model_1_dense_15_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.custom_model_1/dense_15/BiasAdd/ReadVariableOpâ
custom_model_1/dense_15/BiasAddBiasAdd(custom_model_1/dense_15/MatMul:product:06custom_model_1/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
custom_model_1/dense_15/BiasAddò
,custom_model_1/tf.__operators__.add_10/AddV2AddV2(custom_model_1/dense_15/BiasAdd:output:0.custom_model_1/tf.nn.relu_3/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,custom_model_1/tf.__operators__.add_10/AddV2±
 custom_model_1/tf.nn.relu_4/ReluRelu0custom_model_1/tf.__operators__.add_10/AddV2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 custom_model_1/tf.nn.relu_4/Relu×
-custom_model_1/dense_16/MatMul/ReadVariableOpReadVariableOp6custom_model_1_dense_16_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02/
-custom_model_1/dense_16/MatMul/ReadVariableOpä
custom_model_1/dense_16/MatMulMatMul.custom_model_1/tf.nn.relu_4/Relu:activations:05custom_model_1/dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
custom_model_1/dense_16/MatMulÕ
.custom_model_1/dense_16/BiasAdd/ReadVariableOpReadVariableOp7custom_model_1_dense_16_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.custom_model_1/dense_16/BiasAdd/ReadVariableOpâ
custom_model_1/dense_16/BiasAddBiasAdd(custom_model_1/dense_16/MatMul:product:06custom_model_1/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
custom_model_1/dense_16/BiasAddò
,custom_model_1/tf.__operators__.add_11/AddV2AddV2(custom_model_1/dense_16/BiasAdd:output:0.custom_model_1/tf.nn.relu_4/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,custom_model_1/tf.__operators__.add_11/AddV2±
 custom_model_1/tf.nn.relu_5/ReluRelu0custom_model_1/tf.__operators__.add_11/AddV2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 custom_model_1/tf.nn.relu_5/Relu
Acustom_model_1/normalize_1/normalization_1/Reshape/ReadVariableOpReadVariableOpJcustom_model_1_normalize_1_normalization_1_reshape_readvariableop_resource*
_output_shapes	
:*
dtype02C
Acustom_model_1/normalize_1/normalization_1/Reshape/ReadVariableOpÅ
8custom_model_1/normalize_1/normalization_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2:
8custom_model_1/normalize_1/normalization_1/Reshape/shape«
2custom_model_1/normalize_1/normalization_1/ReshapeReshapeIcustom_model_1/normalize_1/normalization_1/Reshape/ReadVariableOp:value:0Acustom_model_1/normalize_1/normalization_1/Reshape/shape:output:0*
T0*
_output_shapes
:	24
2custom_model_1/normalize_1/normalization_1/Reshape
Ccustom_model_1/normalize_1/normalization_1/Reshape_1/ReadVariableOpReadVariableOpLcustom_model_1_normalize_1_normalization_1_reshape_1_readvariableop_resource*
_output_shapes	
:*
dtype02E
Ccustom_model_1/normalize_1/normalization_1/Reshape_1/ReadVariableOpÉ
:custom_model_1/normalize_1/normalization_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2<
:custom_model_1/normalize_1/normalization_1/Reshape_1/shape³
4custom_model_1/normalize_1/normalization_1/Reshape_1ReshapeKcustom_model_1/normalize_1/normalization_1/Reshape_1/ReadVariableOp:value:0Ccustom_model_1/normalize_1/normalization_1/Reshape_1/shape:output:0*
T0*
_output_shapes
:	26
4custom_model_1/normalize_1/normalization_1/Reshape_1
.custom_model_1/normalize_1/normalization_1/subSub.custom_model_1/tf.nn.relu_5/Relu:activations:0;custom_model_1/normalize_1/normalization_1/Reshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.custom_model_1/normalize_1/normalization_1/subÓ
/custom_model_1/normalize_1/normalization_1/SqrtSqrt=custom_model_1/normalize_1/normalization_1/Reshape_1:output:0*
T0*
_output_shapes
:	21
/custom_model_1/normalize_1/normalization_1/Sqrt±
4custom_model_1/normalize_1/normalization_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö326
4custom_model_1/normalize_1/normalization_1/Maximum/y
2custom_model_1/normalize_1/normalization_1/MaximumMaximum3custom_model_1/normalize_1/normalization_1/Sqrt:y:0=custom_model_1/normalize_1/normalization_1/Maximum/y:output:0*
T0*
_output_shapes
:	24
2custom_model_1/normalize_1/normalization_1/Maximum
2custom_model_1/normalize_1/normalization_1/truedivRealDiv2custom_model_1/normalize_1/normalization_1/sub:z:06custom_model_1/normalize_1/normalization_1/Maximum:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2custom_model_1/normalize_1/normalization_1/truedivÖ
-custom_model_1/dense_17/MatMul/ReadVariableOpReadVariableOp6custom_model_1_dense_17_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02/
-custom_model_1/dense_17/MatMul/ReadVariableOpë
custom_model_1/dense_17/MatMulMatMul6custom_model_1/normalize_1/normalization_1/truediv:z:05custom_model_1/dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
custom_model_1/dense_17/MatMulÔ
.custom_model_1/dense_17/BiasAdd/ReadVariableOpReadVariableOp7custom_model_1_dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.custom_model_1/dense_17/BiasAdd/ReadVariableOpá
custom_model_1/dense_17/BiasAddBiasAdd(custom_model_1/dense_17/MatMul:product:06custom_model_1/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
custom_model_1/dense_17/BiasAdd³
IdentityIdentity(custom_model_1/dense_17/BiasAdd:output:0/^custom_model_1/dense_10/BiasAdd/ReadVariableOp.^custom_model_1/dense_10/MatMul/ReadVariableOp/^custom_model_1/dense_11/BiasAdd/ReadVariableOp.^custom_model_1/dense_11/MatMul/ReadVariableOp/^custom_model_1/dense_12/BiasAdd/ReadVariableOp.^custom_model_1/dense_12/MatMul/ReadVariableOp/^custom_model_1/dense_13/BiasAdd/ReadVariableOp.^custom_model_1/dense_13/MatMul/ReadVariableOp/^custom_model_1/dense_14/BiasAdd/ReadVariableOp.^custom_model_1/dense_14/MatMul/ReadVariableOp/^custom_model_1/dense_15/BiasAdd/ReadVariableOp.^custom_model_1/dense_15/MatMul/ReadVariableOp/^custom_model_1/dense_16/BiasAdd/ReadVariableOp.^custom_model_1/dense_16/MatMul/ReadVariableOp/^custom_model_1/dense_17/BiasAdd/ReadVariableOp.^custom_model_1/dense_17/MatMul/ReadVariableOp.^custom_model_1/dense_9/BiasAdd/ReadVariableOp-^custom_model_1/dense_9/MatMul/ReadVariableOp4^custom_model_1/model_2/embedding_6/embedding_lookup4^custom_model_1/model_2/embedding_7/embedding_lookup4^custom_model_1/model_2/embedding_8/embedding_lookup5^custom_model_1/model_3/embedding_10/embedding_lookup5^custom_model_1/model_3/embedding_11/embedding_lookup4^custom_model_1/model_3/embedding_9/embedding_lookupB^custom_model_1/normalize_1/normalization_1/Reshape/ReadVariableOpD^custom_model_1/normalize_1/normalization_1/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*À
_input_shapes®
«:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
: : :::: :::: : ::::::::::::::::::::2`
.custom_model_1/dense_10/BiasAdd/ReadVariableOp.custom_model_1/dense_10/BiasAdd/ReadVariableOp2^
-custom_model_1/dense_10/MatMul/ReadVariableOp-custom_model_1/dense_10/MatMul/ReadVariableOp2`
.custom_model_1/dense_11/BiasAdd/ReadVariableOp.custom_model_1/dense_11/BiasAdd/ReadVariableOp2^
-custom_model_1/dense_11/MatMul/ReadVariableOp-custom_model_1/dense_11/MatMul/ReadVariableOp2`
.custom_model_1/dense_12/BiasAdd/ReadVariableOp.custom_model_1/dense_12/BiasAdd/ReadVariableOp2^
-custom_model_1/dense_12/MatMul/ReadVariableOp-custom_model_1/dense_12/MatMul/ReadVariableOp2`
.custom_model_1/dense_13/BiasAdd/ReadVariableOp.custom_model_1/dense_13/BiasAdd/ReadVariableOp2^
-custom_model_1/dense_13/MatMul/ReadVariableOp-custom_model_1/dense_13/MatMul/ReadVariableOp2`
.custom_model_1/dense_14/BiasAdd/ReadVariableOp.custom_model_1/dense_14/BiasAdd/ReadVariableOp2^
-custom_model_1/dense_14/MatMul/ReadVariableOp-custom_model_1/dense_14/MatMul/ReadVariableOp2`
.custom_model_1/dense_15/BiasAdd/ReadVariableOp.custom_model_1/dense_15/BiasAdd/ReadVariableOp2^
-custom_model_1/dense_15/MatMul/ReadVariableOp-custom_model_1/dense_15/MatMul/ReadVariableOp2`
.custom_model_1/dense_16/BiasAdd/ReadVariableOp.custom_model_1/dense_16/BiasAdd/ReadVariableOp2^
-custom_model_1/dense_16/MatMul/ReadVariableOp-custom_model_1/dense_16/MatMul/ReadVariableOp2`
.custom_model_1/dense_17/BiasAdd/ReadVariableOp.custom_model_1/dense_17/BiasAdd/ReadVariableOp2^
-custom_model_1/dense_17/MatMul/ReadVariableOp-custom_model_1/dense_17/MatMul/ReadVariableOp2^
-custom_model_1/dense_9/BiasAdd/ReadVariableOp-custom_model_1/dense_9/BiasAdd/ReadVariableOp2\
,custom_model_1/dense_9/MatMul/ReadVariableOp,custom_model_1/dense_9/MatMul/ReadVariableOp2j
3custom_model_1/model_2/embedding_6/embedding_lookup3custom_model_1/model_2/embedding_6/embedding_lookup2j
3custom_model_1/model_2/embedding_7/embedding_lookup3custom_model_1/model_2/embedding_7/embedding_lookup2j
3custom_model_1/model_2/embedding_8/embedding_lookup3custom_model_1/model_2/embedding_8/embedding_lookup2l
4custom_model_1/model_3/embedding_10/embedding_lookup4custom_model_1/model_3/embedding_10/embedding_lookup2l
4custom_model_1/model_3/embedding_11/embedding_lookup4custom_model_1/model_3/embedding_11/embedding_lookup2j
3custom_model_1/model_3/embedding_9/embedding_lookup3custom_model_1/model_3/embedding_9/embedding_lookup2
Acustom_model_1/normalize_1/normalization_1/Reshape/ReadVariableOpAcustom_model_1/normalize_1/normalization_1/Reshape/ReadVariableOp2
Ccustom_model_1/normalize_1/normalization_1/Reshape_1/ReadVariableOpCcustom_model_1/normalize_1/normalization_1/Reshape_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namecards0:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namecards1:MI
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


_user_specified_namebets:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
á
|
'__inference_dense_15_layer_call_fn_6414

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_dense_15_layer_call_and_return_conditional_losses_49882
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¹

&__inference_model_2_layer_call_fn_6168

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_model_2_layer_call_and_return_conditional_losses_44922
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 
¹

&__inference_model_2_layer_call_fn_6155

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_model_2_layer_call_and_return_conditional_losses_44472
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 
Ï
q
+__inference_embedding_11_layer_call_fn_6568

inputs
unknown
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_embedding_11_layer_call_and_return_conditional_losses_45412
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù	

E__inference_embedding_7_layer_call_and_return_conditional_losses_6533

inputs
embedding_lookup_6527
identity¢embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Castú
embedding_lookupResourceGatherembedding_lookup_6527Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*(
_class
loc:@embedding_lookup/6527*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_lookupì
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@embedding_lookup/6527*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity¡
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾V
»	
H__inference_custom_model_1_layer_call_and_return_conditional_losses_5283

inputs
inputs_1
inputs_2*
&tf_math_greater_equal_5_greaterequal_y
model_2_5198
model_2_5200
model_2_5202
model_2_5204
model_3_5207
model_3_5209
model_3_5211
model_3_5213.
*tf_clip_by_value_5_clip_by_value_minimum_y&
"tf_clip_by_value_5_clip_by_value_y
dense_9_5225
dense_9_5227
dense_12_5230
dense_12_5232
dense_10_5235
dense_10_5237
dense_11_5240
dense_11_5242
dense_13_5245
dense_13_5247
dense_14_5252
dense_14_5254
dense_15_5258
dense_15_5260
dense_16_5265
dense_16_5267
normalize_1_5272
normalize_1_5274
dense_17_5277
dense_17_5279
identity¢ dense_10/StatefulPartitionedCall¢ dense_11/StatefulPartitionedCall¢ dense_12/StatefulPartitionedCall¢ dense_13/StatefulPartitionedCall¢ dense_14/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall¢ dense_16/StatefulPartitionedCall¢ dense_17/StatefulPartitionedCall¢dense_9/StatefulPartitionedCall¢model_2/StatefulPartitionedCall¢model_3/StatefulPartitionedCall¢#normalize_1/StatefulPartitionedCallÀ
$tf.math.greater_equal_5/GreaterEqualGreaterEqualinputs_2&tf_math_greater_equal_5_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2&
$tf.math.greater_equal_5/GreaterEqual®
model_2/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_2_5198model_2_5200model_2_5202model_2_5204*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_model_2_layer_call_and_return_conditional_losses_44472!
model_2/StatefulPartitionedCall°
model_3/StatefulPartitionedCallStatefulPartitionedCallinputs_1model_3_5207model_3_5209model_3_5211model_3_5213*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_model_3_layer_call_and_return_conditional_losses_46732!
model_3/StatefulPartitionedCallÇ
(tf.clip_by_value_5/clip_by_value/MinimumMinimuminputs_2*tf_clip_by_value_5_clip_by_value_minimum_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2*
(tf.clip_by_value_5/clip_by_value/MinimumÓ
 tf.clip_by_value_5/clip_by_valueMaximum,tf.clip_by_value_5/clip_by_value/Minimum:z:0"tf_clip_by_value_5_clip_by_value_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2"
 tf.clip_by_value_5/clip_by_value
tf.cast_5/CastCast(tf.math.greater_equal_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
tf.cast_5/Castt
tf.concat_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_3/concat/axisæ
tf.concat_3/concatConcatV2(model_2/StatefulPartitionedCall:output:0(model_3/StatefulPartitionedCall:output:0 tf.concat_3/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.concat_3/concat}
tf.concat_4/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
tf.concat_4/concat/axisË
tf.concat_4/concatConcatV2$tf.clip_by_value_5/clip_by_value:z:0tf.cast_5/Cast:y:0 tf.concat_4/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.concat_4/concat¤
dense_9/StatefulPartitionedCallStatefulPartitionedCalltf.concat_3/concat:output:0dense_9_5225dense_9_5227*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_dense_9_layer_call_and_return_conditional_losses_48272!
dense_9/StatefulPartitionedCall©
 dense_12/StatefulPartitionedCallStatefulPartitionedCalltf.concat_4/concat:output:0dense_12_5230dense_12_5232*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_dense_12_layer_call_and_return_conditional_losses_48532"
 dense_12/StatefulPartitionedCall¶
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0dense_10_5235dense_10_5237*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_dense_10_layer_call_and_return_conditional_losses_48802"
 dense_10/StatefulPartitionedCall·
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_5240dense_11_5242*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_dense_11_layer_call_and_return_conditional_losses_49072"
 dense_11/StatefulPartitionedCall·
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_5245dense_13_5247*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_dense_13_layer_call_and_return_conditional_losses_49332"
 dense_13/StatefulPartitionedCall}
tf.concat_5/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
tf.concat_5/concat/axisè
tf.concat_5/concatConcatV2)dense_11/StatefulPartitionedCall:output:0)dense_13/StatefulPartitionedCall:output:0 tf.concat_5/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.concat_5/concat©
 dense_14/StatefulPartitionedCallStatefulPartitionedCalltf.concat_5/concat:output:0dense_14_5252dense_14_5254*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_dense_14_layer_call_and_return_conditional_losses_49612"
 dense_14/StatefulPartitionedCall
tf.nn.relu_3/ReluRelu)dense_14/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.nn.relu_3/Relu­
 dense_15/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_3/Relu:activations:0dense_15_5258dense_15_5260*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_dense_15_layer_call_and_return_conditional_losses_49882"
 dense_15/StatefulPartitionedCallÆ
tf.__operators__.add_10/AddV2AddV2)dense_15/StatefulPartitionedCall:output:0tf.nn.relu_3/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_10/AddV2
tf.nn.relu_4/ReluRelu!tf.__operators__.add_10/AddV2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.nn.relu_4/Relu­
 dense_16/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_4/Relu:activations:0dense_16_5265dense_16_5267*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_dense_16_layer_call_and_return_conditional_losses_50162"
 dense_16/StatefulPartitionedCallÆ
tf.__operators__.add_11/AddV2AddV2)dense_16/StatefulPartitionedCall:output:0tf.nn.relu_4/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_11/AddV2
tf.nn.relu_5/ReluRelu!tf.__operators__.add_11/AddV2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.nn.relu_5/Relu¼
#normalize_1/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_5/Relu:activations:0normalize_1_5272normalize_1_5274*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_normalize_1_layer_call_and_return_conditional_losses_50512%
#normalize_1/StatefulPartitionedCall¹
 dense_17/StatefulPartitionedCallStatefulPartitionedCall,normalize_1/StatefulPartitionedCall:output:0dense_17_5277dense_17_5279*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_dense_17_layer_call_and_return_conditional_losses_50772"
 dense_17/StatefulPartitionedCall¡
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall ^model_2/StatefulPartitionedCall ^model_3/StatefulPartitionedCall$^normalize_1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*À
_input_shapes®
«:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
: : :::: :::: : ::::::::::::::::::::2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2B
model_2/StatefulPartitionedCallmodel_2/StatefulPartitionedCall2B
model_3/StatefulPartitionedCallmodel_3/StatefulPartitionedCall2J
#normalize_1/StatefulPartitionedCall#normalize_1/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
á
|
'__inference_dense_16_layer_call_fn_6433

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_dense_16_layer_call_and_return_conditional_losses_50162
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Û
B__inference_dense_12_layer_call_and_return_conditional_losses_6328

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Û
B__inference_dense_16_layer_call_and_return_conditional_losses_6424

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø
z
*__inference_normalize_1_layer_call_fn_6459
x
unknown
	unknown_0
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_normalize_1_layer_call_and_return_conditional_losses_50512
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
	
Û
B__inference_dense_17_layer_call_and_return_conditional_losses_6469

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Û
B__inference_dense_14_layer_call_and_return_conditional_losses_6386

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ß
|
'__inference_dense_17_layer_call_fn_6478

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_dense_17_layer_call_and_return_conditional_losses_50772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
õ	
Û
B__inference_dense_10_layer_call_and_return_conditional_losses_4880

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
-
Ä
A__inference_model_3_layer_call_and_return_conditional_losses_4673

inputs*
&tf_math_greater_equal_4_greaterequal_y
embedding_11_4655
embedding_9_4658
embedding_10_4663
identity¢$embedding_10/StatefulPartitionedCall¢$embedding_11/StatefulPartitionedCall¢#embedding_9/StatefulPartitionedCallÚ
flatten_3/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_flatten_3_layer_call_and_return_conditional_losses_45132
flatten_3/PartitionedCall
*tf.clip_by_value_4/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_4/clip_by_value/Minimum/yê
(tf.clip_by_value_4/clip_by_value/MinimumMinimum"flatten_3/PartitionedCall:output:03tf.clip_by_value_4/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(tf.clip_by_value_4/clip_by_value/Minimum
"tf.clip_by_value_4/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_4/clip_by_value/yÜ
 tf.clip_by_value_4/clip_by_valueMaximum,tf.clip_by_value_4/clip_by_value/Minimum:z:0+tf.clip_by_value_4/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 tf.clip_by_value_4/clip_by_value
#tf.compat.v1.floor_div_3/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2%
#tf.compat.v1.floor_div_3/FloorDiv/yØ
!tf.compat.v1.floor_div_3/FloorDivFloorDiv$tf.clip_by_value_4/clip_by_value:z:0,tf.compat.v1.floor_div_3/FloorDiv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!tf.compat.v1.floor_div_3/FloorDivÚ
$tf.math.greater_equal_4/GreaterEqualGreaterEqual"flatten_3/PartitionedCall:output:0&tf_math_greater_equal_4_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$tf.math.greater_equal_4/GreaterEqual
tf.math.floormod_3/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2
tf.math.floormod_3/FloorMod/yÆ
tf.math.floormod_3/FloorModFloorMod$tf.clip_by_value_4/clip_by_value:z:0&tf.math.floormod_3/FloorMod/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.floormod_3/FloorModµ
$embedding_11/StatefulPartitionedCallStatefulPartitionedCall$tf.clip_by_value_4/clip_by_value:z:0embedding_11_4655*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_embedding_11_layer_call_and_return_conditional_losses_45412&
$embedding_11/StatefulPartitionedCall²
#embedding_9/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_3/FloorDiv:z:0embedding_9_4658*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_embedding_9_layer_call_and_return_conditional_losses_45632%
#embedding_9/StatefulPartitionedCall
tf.cast_4/CastCast(tf.math.greater_equal_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.cast_4/CastÙ
tf.__operators__.add_8/AddV2AddV2-embedding_11/StatefulPartitionedCall:output:0,embedding_9/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_8/AddV2°
$embedding_10/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_3/FloorMod:z:0embedding_10_4663*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_embedding_10_layer_call_and_return_conditional_losses_45872&
$embedding_10/StatefulPartitionedCallÍ
tf.__operators__.add_9/AddV2AddV2 tf.__operators__.add_8/AddV2:z:0-embedding_10/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_9/AddV2
tf.expand_dims_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
tf.expand_dims_3/ExpandDims/dim¼
tf.expand_dims_3/ExpandDims
ExpandDimstf.cast_4/Cast:y:0(tf.expand_dims_3/ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.expand_dims_3/ExpandDims¶
tf.math.multiply_3/MulMul tf.__operators__.add_9/AddV2:z:0$tf.expand_dims_3/ExpandDims:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.multiply_3/Mul
*tf.math.reduce_sum_3/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_3/Sum/reduction_indices¿
tf.math.reduce_sum_3/SumSumtf.math.multiply_3/Mul:z:03tf.math.reduce_sum_3/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.reduce_sum_3/Sumê
IdentityIdentity!tf.math.reduce_sum_3/Sum:output:0%^embedding_10/StatefulPartitionedCall%^embedding_11/StatefulPartitionedCall$^embedding_9/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: :::2L
$embedding_10/StatefulPartitionedCall$embedding_10/StatefulPartitionedCall2L
$embedding_11/StatefulPartitionedCall$embedding_11/StatefulPartitionedCall2J
#embedding_9/StatefulPartitionedCall#embedding_9/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 
ß
|
'__inference_dense_12_layer_call_fn_6337

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_dense_12_layer_call_and_return_conditional_losses_48532
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*
serving_default
5
bets-
serving_default_bets:0ÿÿÿÿÿÿÿÿÿ

9
cards0/
serving_default_cards0:0ÿÿÿÿÿÿÿÿÿ
9
cards1/
serving_default_cards1:0ÿÿÿÿÿÿÿÿÿ<
dense_170
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:åØ
þ
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-2

layer-9
layer-10
layer_with_weights-3
layer-11
layer_with_weights-4
layer-12
layer_with_weights-5
layer-13
layer_with_weights-6
layer-14
layer-15
layer_with_weights-7
layer-16
layer-17
layer_with_weights-8
layer-18
layer-19
layer-20
layer_with_weights-9
layer-21
layer-22
layer-23
layer_with_weights-10
layer-24
layer_with_weights-11
layer-25
trainable_variables
regularization_losses
	variables
	keras_api

signatures
º_default_save_signature
+»&call_and_return_all_conditional_losses
¼__call__"¢
_tf_keras_network{"class_name": "CustomModel", "name": "custom_model_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "custom_model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards0"}, "name": "cards0", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards1"}, "name": "cards1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "bets"}, "name": "bets", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "model_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_2", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_3", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_3", "inbound_nodes": [["flatten_2", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_2", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_2", "inbound_nodes": [["tf.clip_by_value_3", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_8", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_8", "inbound_nodes": [[["tf.clip_by_value_3", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_6", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_6", "inbound_nodes": [[["tf.compat.v1.floor_div_2", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_2", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_2", "inbound_nodes": [["tf.clip_by_value_3", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_3", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_3", "inbound_nodes": [["flatten_2", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_6", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_6", "inbound_nodes": [["embedding_8", 0, 0, {"y": ["embedding_6", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_7", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_7", "inbound_nodes": [[["tf.math.floormod_2", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_3", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_3", "inbound_nodes": [["tf.math.greater_equal_3", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_7", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_7", "inbound_nodes": [["tf.__operators__.add_6", 0, 0, {"y": ["embedding_7", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_2", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_2", "inbound_nodes": [["tf.cast_3", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_2", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_2", "inbound_nodes": [["tf.__operators__.add_7", 0, 0, {"y": ["tf.expand_dims_2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_2", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_2", "inbound_nodes": [["tf.math.multiply_2", 0, 0, {"axis": 1}]]}], "input_layers": [["input_3", 0, 0]], "output_layers": [["tf.math.reduce_sum_2", 0, 0]]}, "name": "model_2", "inbound_nodes": [[["cards0", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "model_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_3", "inbound_nodes": [[["input_4", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_4", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_4", "inbound_nodes": [["flatten_3", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_3", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_3", "inbound_nodes": [["tf.clip_by_value_4", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_11", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_11", "inbound_nodes": [[["tf.clip_by_value_4", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_9", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_9", "inbound_nodes": [[["tf.compat.v1.floor_div_3", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_3", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_3", "inbound_nodes": [["tf.clip_by_value_4", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_4", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_4", "inbound_nodes": [["flatten_3", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_8", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_8", "inbound_nodes": [["embedding_11", 0, 0, {"y": ["embedding_9", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_10", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_10", "inbound_nodes": [[["tf.math.floormod_3", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_4", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_4", "inbound_nodes": [["tf.math.greater_equal_4", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_9", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_9", "inbound_nodes": [["tf.__operators__.add_8", 0, 0, {"y": ["embedding_10", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_3", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_3", "inbound_nodes": [["tf.cast_4", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_3", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_3", "inbound_nodes": [["tf.__operators__.add_9", 0, 0, {"y": ["tf.expand_dims_3", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_3", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_3", "inbound_nodes": [["tf.math.multiply_3", 0, 0, {"axis": 1}]]}], "input_layers": [["input_4", 0, 0]], "output_layers": [["tf.math.reduce_sum_3", 0, 0]]}, "name": "model_3", "inbound_nodes": [[["cards1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_5", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_5", "inbound_nodes": [["bets", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_3", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_3", "inbound_nodes": [[["model_2", 1, 0, {"axis": 1}], ["model_3", 1, 0, {"axis": 1}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_5", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_5", "inbound_nodes": [["bets", 0, 0, {"clip_value_min": 0.0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_5", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_5", "inbound_nodes": [["tf.math.greater_equal_5", 0, 0, {"dtype": "float32"}]]}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_9", "inbound_nodes": [[["tf.concat_3", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_4", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_4", "inbound_nodes": [[["tf.clip_by_value_5", 0, 0, {"axis": -1}], ["tf.cast_5", 0, 0, {"axis": -1}]]]}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_10", "inbound_nodes": [[["dense_9", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_12", "inbound_nodes": [[["tf.concat_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_11", "inbound_nodes": [[["dense_10", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_13", "inbound_nodes": [[["dense_12", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_5", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_5", "inbound_nodes": [[["dense_11", 0, 0, {"axis": -1}], ["dense_13", 0, 0, {"axis": -1}]]]}, {"class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_14", "inbound_nodes": [[["tf.concat_5", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_3", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_3", "inbound_nodes": [["dense_14", 0, 0, {"name": null}]]}, {"class_name": "Dense", "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_15", "inbound_nodes": [[["tf.nn.relu_3", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_10", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_10", "inbound_nodes": [["dense_15", 0, 0, {"y": ["tf.nn.relu_3", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_4", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_4", "inbound_nodes": [["tf.__operators__.add_10", 0, 0, {"name": null}]]}, {"class_name": "Dense", "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_16", "inbound_nodes": [[["tf.nn.relu_4", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_11", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_11", "inbound_nodes": [["dense_16", 0, 0, {"y": ["tf.nn.relu_4", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_5", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_5", "inbound_nodes": [["tf.__operators__.add_11", 0, 0, {"name": null}]]}, {"class_name": "Normalize", "config": {"name": "normalize_1", "trainable": true, "dtype": "float32"}, "name": "normalize_1", "inbound_nodes": [[["tf.nn.relu_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Constant", "config": {"value": -5}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_17", "inbound_nodes": [[["normalize_1", 0, 0, {}]]]}], "input_layers": [[["cards0", 0, 0], ["cards1", 0, 0]], ["bets", 0, 0]], "output_layers": [["dense_17", 0, 0]]}, "build_input_shape": [[{"class_name": "TensorShape", "items": [null, 2]}, {"class_name": "TensorShape", "items": [null, 3]}], {"class_name": "TensorShape", "items": [null, 10]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "CustomModel", "config": {"name": "custom_model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards0"}, "name": "cards0", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards1"}, "name": "cards1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "bets"}, "name": "bets", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "model_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_2", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_3", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_3", "inbound_nodes": [["flatten_2", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_2", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_2", "inbound_nodes": [["tf.clip_by_value_3", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_8", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_8", "inbound_nodes": [[["tf.clip_by_value_3", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_6", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_6", "inbound_nodes": [[["tf.compat.v1.floor_div_2", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_2", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_2", "inbound_nodes": [["tf.clip_by_value_3", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_3", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_3", "inbound_nodes": [["flatten_2", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_6", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_6", "inbound_nodes": [["embedding_8", 0, 0, {"y": ["embedding_6", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_7", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_7", "inbound_nodes": [[["tf.math.floormod_2", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_3", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_3", "inbound_nodes": [["tf.math.greater_equal_3", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_7", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_7", "inbound_nodes": [["tf.__operators__.add_6", 0, 0, {"y": ["embedding_7", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_2", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_2", "inbound_nodes": [["tf.cast_3", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_2", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_2", "inbound_nodes": [["tf.__operators__.add_7", 0, 0, {"y": ["tf.expand_dims_2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_2", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_2", "inbound_nodes": [["tf.math.multiply_2", 0, 0, {"axis": 1}]]}], "input_layers": [["input_3", 0, 0]], "output_layers": [["tf.math.reduce_sum_2", 0, 0]]}, "name": "model_2", "inbound_nodes": [[["cards0", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "model_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_3", "inbound_nodes": [[["input_4", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_4", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_4", "inbound_nodes": [["flatten_3", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_3", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_3", "inbound_nodes": [["tf.clip_by_value_4", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_11", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_11", "inbound_nodes": [[["tf.clip_by_value_4", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_9", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_9", "inbound_nodes": [[["tf.compat.v1.floor_div_3", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_3", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_3", "inbound_nodes": [["tf.clip_by_value_4", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_4", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_4", "inbound_nodes": [["flatten_3", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_8", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_8", "inbound_nodes": [["embedding_11", 0, 0, {"y": ["embedding_9", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_10", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_10", "inbound_nodes": [[["tf.math.floormod_3", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_4", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_4", "inbound_nodes": [["tf.math.greater_equal_4", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_9", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_9", "inbound_nodes": [["tf.__operators__.add_8", 0, 0, {"y": ["embedding_10", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_3", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_3", "inbound_nodes": [["tf.cast_4", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_3", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_3", "inbound_nodes": [["tf.__operators__.add_9", 0, 0, {"y": ["tf.expand_dims_3", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_3", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_3", "inbound_nodes": [["tf.math.multiply_3", 0, 0, {"axis": 1}]]}], "input_layers": [["input_4", 0, 0]], "output_layers": [["tf.math.reduce_sum_3", 0, 0]]}, "name": "model_3", "inbound_nodes": [[["cards1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_5", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_5", "inbound_nodes": [["bets", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_3", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_3", "inbound_nodes": [[["model_2", 1, 0, {"axis": 1}], ["model_3", 1, 0, {"axis": 1}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_5", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_5", "inbound_nodes": [["bets", 0, 0, {"clip_value_min": 0.0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_5", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_5", "inbound_nodes": [["tf.math.greater_equal_5", 0, 0, {"dtype": "float32"}]]}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_9", "inbound_nodes": [[["tf.concat_3", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_4", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_4", "inbound_nodes": [[["tf.clip_by_value_5", 0, 0, {"axis": -1}], ["tf.cast_5", 0, 0, {"axis": -1}]]]}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_10", "inbound_nodes": [[["dense_9", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_12", "inbound_nodes": [[["tf.concat_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_11", "inbound_nodes": [[["dense_10", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_13", "inbound_nodes": [[["dense_12", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_5", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_5", "inbound_nodes": [[["dense_11", 0, 0, {"axis": -1}], ["dense_13", 0, 0, {"axis": -1}]]]}, {"class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_14", "inbound_nodes": [[["tf.concat_5", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_3", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_3", "inbound_nodes": [["dense_14", 0, 0, {"name": null}]]}, {"class_name": "Dense", "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_15", "inbound_nodes": [[["tf.nn.relu_3", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_10", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_10", "inbound_nodes": [["dense_15", 0, 0, {"y": ["tf.nn.relu_3", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_4", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_4", "inbound_nodes": [["tf.__operators__.add_10", 0, 0, {"name": null}]]}, {"class_name": "Dense", "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_16", "inbound_nodes": [[["tf.nn.relu_4", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_11", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_11", "inbound_nodes": [["dense_16", 0, 0, {"y": ["tf.nn.relu_4", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_5", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_5", "inbound_nodes": [["tf.__operators__.add_11", 0, 0, {"name": null}]]}, {"class_name": "Normalize", "config": {"name": "normalize_1", "trainable": true, "dtype": "float32"}, "name": "normalize_1", "inbound_nodes": [[["tf.nn.relu_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Constant", "config": {"value": -5}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_17", "inbound_nodes": [[["normalize_1", 0, 0, {}]]]}], "input_layers": [[["cards0", 0, 0], ["cards1", 0, 0]], ["bets", 0, 0]], "output_layers": [["dense_17", 0, 0]]}}}
ç"ä
_tf_keras_input_layerÄ{"class_name": "InputLayer", "name": "cards0", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards0"}}
ç"ä
_tf_keras_input_layerÄ{"class_name": "InputLayer", "name": "cards1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards1"}}
å"â
_tf_keras_input_layerÂ{"class_name": "InputLayer", "name": "bets", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "bets"}}
§Q
 layer-0
!layer-1
"layer-2
#layer-3
$layer_with_weights-0
$layer-4
%layer_with_weights-1
%layer-5
&layer-6
'layer-7
(layer-8
)layer_with_weights-2
)layer-9
*layer-10
+layer-11
,layer-12
-layer-13
.layer-14
/trainable_variables
0regularization_losses
1	variables
2	keras_api
+½&call_and_return_all_conditional_losses
¾__call__"N
_tf_keras_networkäM{"class_name": "Functional", "name": "model_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_2", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_3", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_3", "inbound_nodes": [["flatten_2", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_2", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_2", "inbound_nodes": [["tf.clip_by_value_3", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_8", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_8", "inbound_nodes": [[["tf.clip_by_value_3", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_6", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_6", "inbound_nodes": [[["tf.compat.v1.floor_div_2", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_2", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_2", "inbound_nodes": [["tf.clip_by_value_3", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_3", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_3", "inbound_nodes": [["flatten_2", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_6", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_6", "inbound_nodes": [["embedding_8", 0, 0, {"y": ["embedding_6", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_7", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_7", "inbound_nodes": [[["tf.math.floormod_2", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_3", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_3", "inbound_nodes": [["tf.math.greater_equal_3", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_7", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_7", "inbound_nodes": [["tf.__operators__.add_6", 0, 0, {"y": ["embedding_7", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_2", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_2", "inbound_nodes": [["tf.cast_3", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_2", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_2", "inbound_nodes": [["tf.__operators__.add_7", 0, 0, {"y": ["tf.expand_dims_2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_2", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_2", "inbound_nodes": [["tf.math.multiply_2", 0, 0, {"axis": 1}]]}], "input_layers": [["input_3", 0, 0]], "output_layers": [["tf.math.reduce_sum_2", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 2]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_2", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_3", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_3", "inbound_nodes": [["flatten_2", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_2", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_2", "inbound_nodes": [["tf.clip_by_value_3", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_8", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_8", "inbound_nodes": [[["tf.clip_by_value_3", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_6", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_6", "inbound_nodes": [[["tf.compat.v1.floor_div_2", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_2", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_2", "inbound_nodes": [["tf.clip_by_value_3", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_3", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_3", "inbound_nodes": [["flatten_2", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_6", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_6", "inbound_nodes": [["embedding_8", 0, 0, {"y": ["embedding_6", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_7", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_7", "inbound_nodes": [[["tf.math.floormod_2", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_3", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_3", "inbound_nodes": [["tf.math.greater_equal_3", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_7", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_7", "inbound_nodes": [["tf.__operators__.add_6", 0, 0, {"y": ["embedding_7", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_2", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_2", "inbound_nodes": [["tf.cast_3", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_2", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_2", "inbound_nodes": [["tf.__operators__.add_7", 0, 0, {"y": ["tf.expand_dims_2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_2", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_2", "inbound_nodes": [["tf.math.multiply_2", 0, 0, {"axis": 1}]]}], "input_layers": [["input_3", 0, 0]], "output_layers": [["tf.math.reduce_sum_2", 0, 0]]}}}
³Q
3layer-0
4layer-1
5layer-2
6layer-3
7layer_with_weights-0
7layer-4
8layer_with_weights-1
8layer-5
9layer-6
:layer-7
;layer-8
<layer_with_weights-2
<layer-9
=layer-10
>layer-11
?layer-12
@layer-13
Alayer-14
Btrainable_variables
Cregularization_losses
D	variables
E	keras_api
+¿&call_and_return_all_conditional_losses
À__call__"N
_tf_keras_networkðM{"class_name": "Functional", "name": "model_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_3", "inbound_nodes": [[["input_4", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_4", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_4", "inbound_nodes": [["flatten_3", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_3", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_3", "inbound_nodes": [["tf.clip_by_value_4", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_11", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_11", "inbound_nodes": [[["tf.clip_by_value_4", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_9", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_9", "inbound_nodes": [[["tf.compat.v1.floor_div_3", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_3", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_3", "inbound_nodes": [["tf.clip_by_value_4", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_4", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_4", "inbound_nodes": [["flatten_3", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_8", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_8", "inbound_nodes": [["embedding_11", 0, 0, {"y": ["embedding_9", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_10", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_10", "inbound_nodes": [[["tf.math.floormod_3", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_4", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_4", "inbound_nodes": [["tf.math.greater_equal_4", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_9", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_9", "inbound_nodes": [["tf.__operators__.add_8", 0, 0, {"y": ["embedding_10", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_3", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_3", "inbound_nodes": [["tf.cast_4", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_3", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_3", "inbound_nodes": [["tf.__operators__.add_9", 0, 0, {"y": ["tf.expand_dims_3", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_3", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_3", "inbound_nodes": [["tf.math.multiply_3", 0, 0, {"axis": 1}]]}], "input_layers": [["input_4", 0, 0]], "output_layers": [["tf.math.reduce_sum_3", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 3]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_3", "inbound_nodes": [[["input_4", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_4", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_4", "inbound_nodes": [["flatten_3", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_3", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_3", "inbound_nodes": [["tf.clip_by_value_4", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_11", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_11", "inbound_nodes": [[["tf.clip_by_value_4", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_9", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_9", "inbound_nodes": [[["tf.compat.v1.floor_div_3", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_3", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_3", "inbound_nodes": [["tf.clip_by_value_4", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_4", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_4", "inbound_nodes": [["flatten_3", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_8", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_8", "inbound_nodes": [["embedding_11", 0, 0, {"y": ["embedding_9", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_10", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_10", "inbound_nodes": [[["tf.math.floormod_3", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_4", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_4", "inbound_nodes": [["tf.math.greater_equal_4", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_9", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_9", "inbound_nodes": [["tf.__operators__.add_8", 0, 0, {"y": ["embedding_10", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_3", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_3", "inbound_nodes": [["tf.cast_4", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_3", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_3", "inbound_nodes": [["tf.__operators__.add_9", 0, 0, {"y": ["tf.expand_dims_3", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_3", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_3", "inbound_nodes": [["tf.math.multiply_3", 0, 0, {"axis": 1}]]}], "input_layers": [["input_4", 0, 0]], "output_layers": [["tf.math.reduce_sum_3", 0, 0]]}}}
ù
F	keras_api"ç
_tf_keras_layerÍ{"class_name": "TFOpLambda", "name": "tf.math.greater_equal_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.greater_equal_5", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}}
Õ
G	keras_api"Ã
_tf_keras_layer©{"class_name": "TFOpLambda", "name": "tf.concat_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.concat_3", "trainable": true, "dtype": "float32", "function": "concat"}}
ê
H	keras_api"Ø
_tf_keras_layer¾{"class_name": "TFOpLambda", "name": "tf.clip_by_value_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.clip_by_value_5", "trainable": true, "dtype": "float32", "function": "clip_by_value"}}
Ï
I	keras_api"½
_tf_keras_layer£{"class_name": "TFOpLambda", "name": "tf.cast_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.cast_5", "trainable": true, "dtype": "float32", "function": "cast"}}
õ

Jkernel
Kbias
Ltrainable_variables
Mregularization_losses
N	variables
O	keras_api
+Á&call_and_return_all_conditional_losses
Â__call__"Î
_tf_keras_layer´{"class_name": "Dense", "name": "dense_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
Õ
P	keras_api"Ã
_tf_keras_layer©{"class_name": "TFOpLambda", "name": "tf.concat_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.concat_4", "trainable": true, "dtype": "float32", "function": "concat"}}
÷

Qkernel
Rbias
Strainable_variables
Tregularization_losses
U	variables
V	keras_api
+Ã&call_and_return_all_conditional_losses
Ä__call__"Ð
_tf_keras_layer¶{"class_name": "Dense", "name": "dense_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
÷

Wkernel
Xbias
Ytrainable_variables
Zregularization_losses
[	variables
\	keras_api
+Å&call_and_return_all_conditional_losses
Æ__call__"Ð
_tf_keras_layer¶{"class_name": "Dense", "name": "dense_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20]}}
÷

]kernel
^bias
_trainable_variables
`regularization_losses
a	variables
b	keras_api
+Ç&call_and_return_all_conditional_losses
È__call__"Ð
_tf_keras_layer¶{"class_name": "Dense", "name": "dense_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
ù

ckernel
dbias
etrainable_variables
fregularization_losses
g	variables
h	keras_api
+É&call_and_return_all_conditional_losses
Ê__call__"Ò
_tf_keras_layer¸{"class_name": "Dense", "name": "dense_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
Õ
i	keras_api"Ã
_tf_keras_layer©{"class_name": "TFOpLambda", "name": "tf.concat_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.concat_5", "trainable": true, "dtype": "float32", "function": "concat"}}
ù

jkernel
kbias
ltrainable_variables
mregularization_losses
n	variables
o	keras_api
+Ë&call_and_return_all_conditional_losses
Ì__call__"Ò
_tf_keras_layer¸{"class_name": "Dense", "name": "dense_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
Ø
p	keras_api"Æ
_tf_keras_layer¬{"class_name": "TFOpLambda", "name": "tf.nn.relu_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.nn.relu_3", "trainable": true, "dtype": "float32", "function": "nn.relu"}}
ù

qkernel
rbias
strainable_variables
tregularization_losses
u	variables
v	keras_api
+Í&call_and_return_all_conditional_losses
Î__call__"Ò
_tf_keras_layer¸{"class_name": "Dense", "name": "dense_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
ø
w	keras_api"æ
_tf_keras_layerÌ{"class_name": "TFOpLambda", "name": "tf.__operators__.add_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_10", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
Ø
x	keras_api"Æ
_tf_keras_layer¬{"class_name": "TFOpLambda", "name": "tf.nn.relu_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.nn.relu_4", "trainable": true, "dtype": "float32", "function": "nn.relu"}}
ù

ykernel
zbias
{trainable_variables
|regularization_losses
}	variables
~	keras_api
+Ï&call_and_return_all_conditional_losses
Ð__call__"Ò
_tf_keras_layer¸{"class_name": "Dense", "name": "dense_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
ø
	keras_api"æ
_tf_keras_layerÌ{"class_name": "TFOpLambda", "name": "tf.__operators__.add_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_11", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
Ù
	keras_api"Æ
_tf_keras_layer¬{"class_name": "TFOpLambda", "name": "tf.nn.relu_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.nn.relu_5", "trainable": true, "dtype": "float32", "function": "nn.relu"}}
Ò
	normalize
trainable_variables
regularization_losses
	variables
	keras_api
+Ñ&call_and_return_all_conditional_losses
Ò__call__"­
_tf_keras_layer{"class_name": "Normalize", "name": "normalize_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "normalize_1", "trainable": true, "dtype": "float32"}}

kernel
	bias
trainable_variables
regularization_losses
	variables
	keras_api
+Ó&call_and_return_all_conditional_losses
Ô__call__"Þ
_tf_keras_layerÄ{"class_name": "Dense", "name": "dense_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Constant", "config": {"value": -5}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
Þ
0
1
2
3
4
5
J6
K7
Q8
R9
W10
X11
]12
^13
c14
d15
j16
k17
q18
r19
y20
z21
22
23"
trackable_list_wrapper
 "
trackable_list_wrapper
ù
0
1
2
3
4
5
J6
K7
Q8
R9
W10
X11
]12
^13
c14
d15
j16
k17
q18
r19
y20
z21
22
23
24
25
26"
trackable_list_wrapper
Ó
layers
metrics
 layer_regularization_losses
trainable_variables
layer_metrics
regularization_losses
non_trainable_variables
	variables
¼__call__
º_default_save_signature
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses"
_generic_user_object
-
Õserving_default"
signature_map
é"æ
_tf_keras_input_layerÆ{"class_name": "InputLayer", "name": "input_3", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}}
ì
trainable_variables
regularization_losses
	variables
	keras_api
+Ö&call_and_return_all_conditional_losses
×__call__"×
_tf_keras_layer½{"class_name": "Flatten", "name": "flatten_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ë
	keras_api"Ø
_tf_keras_layer¾{"class_name": "TFOpLambda", "name": "tf.clip_by_value_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.clip_by_value_3", "trainable": true, "dtype": "float32", "function": "clip_by_value"}}
ý
	keras_api"ê
_tf_keras_layerÐ{"class_name": "TFOpLambda", "name": "tf.compat.v1.floor_div_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.compat.v1.floor_div_2", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}}
³

embeddings
 trainable_variables
¡regularization_losses
¢	variables
£	keras_api
+Ø&call_and_return_all_conditional_losses
Ù__call__"
_tf_keras_layeró{"class_name": "Embedding", "name": "embedding_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_8", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}}
³

embeddings
¤trainable_variables
¥regularization_losses
¦	variables
§	keras_api
+Ú&call_and_return_all_conditional_losses
Û__call__"
_tf_keras_layeró{"class_name": "Embedding", "name": "embedding_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_6", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}}
ë
¨	keras_api"Ø
_tf_keras_layer¾{"class_name": "TFOpLambda", "name": "tf.math.floormod_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.floormod_2", "trainable": true, "dtype": "float32", "function": "math.floormod"}}
ú
©	keras_api"ç
_tf_keras_layerÍ{"class_name": "TFOpLambda", "name": "tf.math.greater_equal_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.greater_equal_3", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}}
÷
ª	keras_api"ä
_tf_keras_layerÊ{"class_name": "TFOpLambda", "name": "tf.__operators__.add_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_6", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
²

embeddings
«trainable_variables
¬regularization_losses
­	variables
®	keras_api
+Ü&call_and_return_all_conditional_losses
Ý__call__"
_tf_keras_layerò{"class_name": "Embedding", "name": "embedding_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_7", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}}
Ð
¯	keras_api"½
_tf_keras_layer£{"class_name": "TFOpLambda", "name": "tf.cast_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.cast_3", "trainable": true, "dtype": "float32", "function": "cast"}}
÷
°	keras_api"ä
_tf_keras_layerÊ{"class_name": "TFOpLambda", "name": "tf.__operators__.add_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_7", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
å
±	keras_api"Ò
_tf_keras_layer¸{"class_name": "TFOpLambda", "name": "tf.expand_dims_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.expand_dims_2", "trainable": true, "dtype": "float32", "function": "expand_dims"}}
ë
²	keras_api"Ø
_tf_keras_layer¾{"class_name": "TFOpLambda", "name": "tf.math.multiply_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_2", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
ñ
³	keras_api"Þ
_tf_keras_layerÄ{"class_name": "TFOpLambda", "name": "tf.math.reduce_sum_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.reduce_sum_2", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}}
8
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
8
0
1
2"
trackable_list_wrapper
µ
´layers
µmetrics
 ¶layer_regularization_losses
/trainable_variables
·layer_metrics
0regularization_losses
¸non_trainable_variables
1	variables
¾__call__
+½&call_and_return_all_conditional_losses
'½"call_and_return_conditional_losses"
_generic_user_object
é"æ
_tf_keras_input_layerÆ{"class_name": "InputLayer", "name": "input_4", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}}
ì
¹trainable_variables
ºregularization_losses
»	variables
¼	keras_api
+Þ&call_and_return_all_conditional_losses
ß__call__"×
_tf_keras_layer½{"class_name": "Flatten", "name": "flatten_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ë
½	keras_api"Ø
_tf_keras_layer¾{"class_name": "TFOpLambda", "name": "tf.clip_by_value_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.clip_by_value_4", "trainable": true, "dtype": "float32", "function": "clip_by_value"}}
ý
¾	keras_api"ê
_tf_keras_layerÐ{"class_name": "TFOpLambda", "name": "tf.compat.v1.floor_div_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.compat.v1.floor_div_3", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}}
µ

embeddings
¿trainable_variables
Àregularization_losses
Á	variables
Â	keras_api
+à&call_and_return_all_conditional_losses
á__call__"
_tf_keras_layerõ{"class_name": "Embedding", "name": "embedding_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_11", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}}
³

embeddings
Ãtrainable_variables
Äregularization_losses
Å	variables
Æ	keras_api
+â&call_and_return_all_conditional_losses
ã__call__"
_tf_keras_layeró{"class_name": "Embedding", "name": "embedding_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_9", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}}
ë
Ç	keras_api"Ø
_tf_keras_layer¾{"class_name": "TFOpLambda", "name": "tf.math.floormod_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.floormod_3", "trainable": true, "dtype": "float32", "function": "math.floormod"}}
ú
È	keras_api"ç
_tf_keras_layerÍ{"class_name": "TFOpLambda", "name": "tf.math.greater_equal_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.greater_equal_4", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}}
÷
É	keras_api"ä
_tf_keras_layerÊ{"class_name": "TFOpLambda", "name": "tf.__operators__.add_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_8", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
´

embeddings
Êtrainable_variables
Ëregularization_losses
Ì	variables
Í	keras_api
+ä&call_and_return_all_conditional_losses
å__call__"
_tf_keras_layerô{"class_name": "Embedding", "name": "embedding_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_10", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}}
Ð
Î	keras_api"½
_tf_keras_layer£{"class_name": "TFOpLambda", "name": "tf.cast_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.cast_4", "trainable": true, "dtype": "float32", "function": "cast"}}
÷
Ï	keras_api"ä
_tf_keras_layerÊ{"class_name": "TFOpLambda", "name": "tf.__operators__.add_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_9", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
å
Ð	keras_api"Ò
_tf_keras_layer¸{"class_name": "TFOpLambda", "name": "tf.expand_dims_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.expand_dims_3", "trainable": true, "dtype": "float32", "function": "expand_dims"}}
ë
Ñ	keras_api"Ø
_tf_keras_layer¾{"class_name": "TFOpLambda", "name": "tf.math.multiply_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_3", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
ñ
Ò	keras_api"Þ
_tf_keras_layerÄ{"class_name": "TFOpLambda", "name": "tf.math.reduce_sum_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.reduce_sum_3", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}}
8
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
8
0
1
2"
trackable_list_wrapper
µ
Ólayers
Ômetrics
 Õlayer_regularization_losses
Btrainable_variables
Ölayer_metrics
Cregularization_losses
×non_trainable_variables
D	variables
À__call__
+¿&call_and_return_all_conditional_losses
'¿"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
": 
2dense_9/kernel
:2dense_9/bias
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
µ
Ølayers
Ùmetrics
 Úlayer_regularization_losses
Ltrainable_variables
Ûlayer_metrics
Mregularization_losses
Ünon_trainable_variables
N	variables
Â__call__
+Á&call_and_return_all_conditional_losses
'Á"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
#:!
2dense_10/kernel
:2dense_10/bias
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
µ
Ýlayers
Þmetrics
 ßlayer_regularization_losses
Strainable_variables
àlayer_metrics
Tregularization_losses
ánon_trainable_variables
U	variables
Ä__call__
+Ã&call_and_return_all_conditional_losses
'Ã"call_and_return_conditional_losses"
_generic_user_object
": 	2dense_12/kernel
:2dense_12/bias
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
µ
âlayers
ãmetrics
 älayer_regularization_losses
Ytrainable_variables
ålayer_metrics
Zregularization_losses
ænon_trainable_variables
[	variables
Æ__call__
+Å&call_and_return_all_conditional_losses
'Å"call_and_return_conditional_losses"
_generic_user_object
#:!
2dense_11/kernel
:2dense_11/bias
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
µ
çlayers
èmetrics
 élayer_regularization_losses
_trainable_variables
êlayer_metrics
`regularization_losses
ënon_trainable_variables
a	variables
È__call__
+Ç&call_and_return_all_conditional_losses
'Ç"call_and_return_conditional_losses"
_generic_user_object
#:!
2dense_13/kernel
:2dense_13/bias
.
c0
d1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
µ
ìlayers
ímetrics
 îlayer_regularization_losses
etrainable_variables
ïlayer_metrics
fregularization_losses
ðnon_trainable_variables
g	variables
Ê__call__
+É&call_and_return_all_conditional_losses
'É"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
#:!
2dense_14/kernel
:2dense_14/bias
.
j0
k1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
µ
ñlayers
òmetrics
 ólayer_regularization_losses
ltrainable_variables
ôlayer_metrics
mregularization_losses
õnon_trainable_variables
n	variables
Ì__call__
+Ë&call_and_return_all_conditional_losses
'Ë"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
#:!
2dense_15/kernel
:2dense_15/bias
.
q0
r1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
µ
ölayers
÷metrics
 ølayer_regularization_losses
strainable_variables
ùlayer_metrics
tregularization_losses
únon_trainable_variables
u	variables
Î__call__
+Í&call_and_return_all_conditional_losses
'Í"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
#:!
2dense_16/kernel
:2dense_16/bias
.
y0
z1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
µ
ûlayers
ümetrics
 ýlayer_regularization_losses
{trainable_variables
þlayer_metrics
|regularization_losses
ÿnon_trainable_variables
}	variables
Ð__call__
+Ï&call_and_return_all_conditional_losses
'Ï"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object

state_variables
_broadcast_shape
	mean
variance

count
	keras_api"¶
_tf_keras_layer{"class_name": "Normalization", "name": "normalization_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "normalization_1", "trainable": true, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [-1]}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
8
0
1
2"
trackable_list_wrapper
¸
layers
metrics
 layer_regularization_losses
trainable_variables
layer_metrics
regularization_losses
non_trainable_variables
	variables
Ò__call__
+Ñ&call_and_return_all_conditional_losses
'Ñ"call_and_return_conditional_losses"
_generic_user_object
": 	2dense_17/kernel
:2dense_17/bias
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
layers
metrics
 layer_regularization_losses
trainable_variables
layer_metrics
regularization_losses
non_trainable_variables
	variables
Ô__call__
+Ó&call_and_return_all_conditional_losses
'Ó"call_and_return_conditional_losses"
_generic_user_object
):'	42embedding_8/embeddings
):'	2embedding_6/embeddings
):'	2embedding_7/embeddings
*:(	42embedding_11/embeddings
):'	2embedding_9/embeddings
*:(	2embedding_10/embeddings
-:+2 normalize_1/normalization_1/mean
1:/2$normalize_1/normalization_1/variance
):'	 2!normalize_1/normalization_1/count
æ
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25"
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
8
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
layers
metrics
 layer_regularization_losses
trainable_variables
layer_metrics
regularization_losses
non_trainable_variables
	variables
×__call__
+Ö&call_and_return_all_conditional_losses
'Ö"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
¸
layers
metrics
 layer_regularization_losses
 trainable_variables
layer_metrics
¡regularization_losses
non_trainable_variables
¢	variables
Ù__call__
+Ø&call_and_return_all_conditional_losses
'Ø"call_and_return_conditional_losses"
_generic_user_object
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
¸
layers
metrics
 layer_regularization_losses
¤trainable_variables
layer_metrics
¥regularization_losses
non_trainable_variables
¦	variables
Û__call__
+Ú&call_and_return_all_conditional_losses
'Ú"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
¸
layers
metrics
 layer_regularization_losses
«trainable_variables
 layer_metrics
¬regularization_losses
¡non_trainable_variables
­	variables
Ý__call__
+Ü&call_and_return_all_conditional_losses
'Ü"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object

 0
!1
"2
#3
$4
%5
&6
'7
(8
)9
*10
+11
,12
-13
.14"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¢layers
£metrics
 ¤layer_regularization_losses
¹trainable_variables
¥layer_metrics
ºregularization_losses
¦non_trainable_variables
»	variables
ß__call__
+Þ&call_and_return_all_conditional_losses
'Þ"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
¸
§layers
¨metrics
 ©layer_regularization_losses
¿trainable_variables
ªlayer_metrics
Àregularization_losses
«non_trainable_variables
Á	variables
á__call__
+à&call_and_return_all_conditional_losses
'à"call_and_return_conditional_losses"
_generic_user_object
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
¸
¬layers
­metrics
 ®layer_regularization_losses
Ãtrainable_variables
¯layer_metrics
Äregularization_losses
°non_trainable_variables
Å	variables
ã__call__
+â&call_and_return_all_conditional_losses
'â"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
¸
±layers
²metrics
 ³layer_regularization_losses
Êtrainable_variables
´layer_metrics
Ëregularization_losses
µnon_trainable_variables
Ì	variables
å__call__
+ä&call_and_return_all_conditional_losses
'ä"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object

30
41
52
63
74
85
96
:7
;8
<9
=10
>11
?12
@13
A14"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
F
	mean
variance

count"
trackable_dict_wrapper
 "
trackable_list_wrapper
"
_generic_user_object
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
8
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
¿

¶total

·count
¸	variables
¹	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
¶0
·1"
trackable_list_wrapper
.
¸	variables"
_generic_user_object
¨2¥
__inference__wrapped_model_4277
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *q¢n
li
GD
 
cards0ÿÿÿÿÿÿÿÿÿ
 
cards1ÿÿÿÿÿÿÿÿÿ

betsÿÿÿÿÿÿÿÿÿ

î2ë
H__inference_custom_model_1_layer_call_and_return_conditional_losses_5750
H__inference_custom_model_1_layer_call_and_return_conditional_losses_5920
H__inference_custom_model_1_layer_call_and_return_conditional_losses_5186
H__inference_custom_model_1_layer_call_and_return_conditional_losses_5094À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2ÿ
-__inference_custom_model_1_layer_call_fn_5989
-__inference_custom_model_1_layer_call_fn_5348
-__inference_custom_model_1_layer_call_fn_5509
-__inference_custom_model_1_layer_call_fn_6058À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ò2Ï
A__inference_model_2_layer_call_and_return_conditional_losses_4412
A__inference_model_2_layer_call_and_return_conditional_losses_6100
A__inference_model_2_layer_call_and_return_conditional_losses_6142
A__inference_model_2_layer_call_and_return_conditional_losses_4380À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
æ2ã
&__inference_model_2_layer_call_fn_6168
&__inference_model_2_layer_call_fn_4458
&__inference_model_2_layer_call_fn_4503
&__inference_model_2_layer_call_fn_6155À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ò2Ï
A__inference_model_3_layer_call_and_return_conditional_losses_6252
A__inference_model_3_layer_call_and_return_conditional_losses_4638
A__inference_model_3_layer_call_and_return_conditional_losses_6210
A__inference_model_3_layer_call_and_return_conditional_losses_4606À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
æ2ã
&__inference_model_3_layer_call_fn_6265
&__inference_model_3_layer_call_fn_6278
&__inference_model_3_layer_call_fn_4684
&__inference_model_3_layer_call_fn_4729À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ë2è
A__inference_dense_9_layer_call_and_return_conditional_losses_6289¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ð2Í
&__inference_dense_9_layer_call_fn_6298¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_dense_10_layer_call_and_return_conditional_losses_6309¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_dense_10_layer_call_fn_6318¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_dense_12_layer_call_and_return_conditional_losses_6328¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_dense_12_layer_call_fn_6337¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_dense_11_layer_call_and_return_conditional_losses_6348¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_dense_11_layer_call_fn_6357¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_dense_13_layer_call_and_return_conditional_losses_6367¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_dense_13_layer_call_fn_6376¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_dense_14_layer_call_and_return_conditional_losses_6386¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_dense_14_layer_call_fn_6395¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_dense_15_layer_call_and_return_conditional_losses_6405¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_dense_15_layer_call_fn_6414¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_dense_16_layer_call_and_return_conditional_losses_6424¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_dense_16_layer_call_fn_6433¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ê2ç
E__inference_normalize_1_layer_call_and_return_conditional_losses_6450
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ï2Ì
*__inference_normalize_1_layer_call_fn_6459
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_dense_17_layer_call_and_return_conditional_losses_6469¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_dense_17_layer_call_fn_6478¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÔBÑ
"__inference_signature_wrapper_5580betscards0cards1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_flatten_2_layer_call_and_return_conditional_losses_6484¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_flatten_2_layer_call_fn_6489¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_embedding_8_layer_call_and_return_conditional_losses_6499¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_embedding_8_layer_call_fn_6506¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_embedding_6_layer_call_and_return_conditional_losses_6516¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_embedding_6_layer_call_fn_6523¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_embedding_7_layer_call_and_return_conditional_losses_6533¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_embedding_7_layer_call_fn_6540¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_flatten_3_layer_call_and_return_conditional_losses_6546¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_flatten_3_layer_call_fn_6551¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_embedding_11_layer_call_and_return_conditional_losses_6561¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_embedding_11_layer_call_fn_6568¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_embedding_9_layer_call_and_return_conditional_losses_6578¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_embedding_9_layer_call_fn_6585¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_embedding_10_layer_call_and_return_conditional_losses_6595¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_embedding_10_layer_call_fn_6602¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
	J
Const
J	
Const_1
J	
Const_2
J	
Const_3
J	
Const_4
__inference__wrapped_model_4277â.æçèéêJKWXQR]^cdjkqryz{¢x
q¢n
li
GD
 
cards0ÿÿÿÿÿÿÿÿÿ
 
cards1ÿÿÿÿÿÿÿÿÿ

betsÿÿÿÿÿÿÿÿÿ

ª "3ª0
.
dense_17"
dense_17ÿÿÿÿÿÿÿÿÿ«
H__inference_custom_model_1_layer_call_and_return_conditional_losses_5094Þ.æçèéêJKWXQR]^cdjkqryz¢
y¢v
li
GD
 
cards0ÿÿÿÿÿÿÿÿÿ
 
cards1ÿÿÿÿÿÿÿÿÿ

betsÿÿÿÿÿÿÿÿÿ

p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 «
H__inference_custom_model_1_layer_call_and_return_conditional_losses_5186Þ.æçèéêJKWXQR]^cdjkqryz¢
y¢v
li
GD
 
cards0ÿÿÿÿÿÿÿÿÿ
 
cards1ÿÿÿÿÿÿÿÿÿ

betsÿÿÿÿÿÿÿÿÿ

p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¹
H__inference_custom_model_1_layer_call_and_return_conditional_losses_5750ì.æçèéêJKWXQR]^cdjkqryz¢
¢
xu
OL
$!

inputs/0/0ÿÿÿÿÿÿÿÿÿ
$!

inputs/0/1ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ

p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¹
H__inference_custom_model_1_layer_call_and_return_conditional_losses_5920ì.æçèéêJKWXQR]^cdjkqryz¢
¢
xu
OL
$!

inputs/0/0ÿÿÿÿÿÿÿÿÿ
$!

inputs/0/1ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ

p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
-__inference_custom_model_1_layer_call_fn_5348Ñ.æçèéêJKWXQR]^cdjkqryz¢
y¢v
li
GD
 
cards0ÿÿÿÿÿÿÿÿÿ
 
cards1ÿÿÿÿÿÿÿÿÿ

betsÿÿÿÿÿÿÿÿÿ

p

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_custom_model_1_layer_call_fn_5509Ñ.æçèéêJKWXQR]^cdjkqryz¢
y¢v
li
GD
 
cards0ÿÿÿÿÿÿÿÿÿ
 
cards1ÿÿÿÿÿÿÿÿÿ

betsÿÿÿÿÿÿÿÿÿ

p 

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_custom_model_1_layer_call_fn_5989ß.æçèéêJKWXQR]^cdjkqryz¢
¢
xu
OL
$!

inputs/0/0ÿÿÿÿÿÿÿÿÿ
$!

inputs/0/1ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ

p

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_custom_model_1_layer_call_fn_6058ß.æçèéêJKWXQR]^cdjkqryz¢
¢
xu
OL
$!

inputs/0/0ÿÿÿÿÿÿÿÿÿ
$!

inputs/0/1ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ

p 

 
ª "ÿÿÿÿÿÿÿÿÿ¤
B__inference_dense_10_layer_call_and_return_conditional_losses_6309^QR0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 |
'__inference_dense_10_layer_call_fn_6318QQR0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¤
B__inference_dense_11_layer_call_and_return_conditional_losses_6348^]^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 |
'__inference_dense_11_layer_call_fn_6357Q]^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ£
B__inference_dense_12_layer_call_and_return_conditional_losses_6328]WX/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 {
'__inference_dense_12_layer_call_fn_6337PWX/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¤
B__inference_dense_13_layer_call_and_return_conditional_losses_6367^cd0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 |
'__inference_dense_13_layer_call_fn_6376Qcd0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¤
B__inference_dense_14_layer_call_and_return_conditional_losses_6386^jk0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 |
'__inference_dense_14_layer_call_fn_6395Qjk0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¤
B__inference_dense_15_layer_call_and_return_conditional_losses_6405^qr0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 |
'__inference_dense_15_layer_call_fn_6414Qqr0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¤
B__inference_dense_16_layer_call_and_return_conditional_losses_6424^yz0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 |
'__inference_dense_16_layer_call_fn_6433Qyz0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¥
B__inference_dense_17_layer_call_and_return_conditional_losses_6469_0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
'__inference_dense_17_layer_call_fn_6478R0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ£
A__inference_dense_9_layer_call_and_return_conditional_losses_6289^JK0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 {
&__inference_dense_9_layer_call_fn_6298QJK0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ«
F__inference_embedding_10_layer_call_and_return_conditional_losses_6595a/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 
+__inference_embedding_10_layer_call_fn_6602T/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ«
F__inference_embedding_11_layer_call_and_return_conditional_losses_6561a/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 
+__inference_embedding_11_layer_call_fn_6568T/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿª
E__inference_embedding_6_layer_call_and_return_conditional_losses_6516a/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 
*__inference_embedding_6_layer_call_fn_6523T/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿª
E__inference_embedding_7_layer_call_and_return_conditional_losses_6533a/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 
*__inference_embedding_7_layer_call_fn_6540T/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿª
E__inference_embedding_8_layer_call_and_return_conditional_losses_6499a/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 
*__inference_embedding_8_layer_call_fn_6506T/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿª
E__inference_embedding_9_layer_call_and_return_conditional_losses_6578a/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 
*__inference_embedding_9_layer_call_fn_6585T/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
C__inference_flatten_2_layer_call_and_return_conditional_losses_6484X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 w
(__inference_flatten_2_layer_call_fn_6489K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
C__inference_flatten_3_layer_call_and_return_conditional_losses_6546X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 w
(__inference_flatten_3_layer_call_fn_6551K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ±
A__inference_model_2_layer_call_and_return_conditional_losses_4380lç8¢5
.¢+
!
input_3ÿÿÿÿÿÿÿÿÿ
p

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ±
A__inference_model_2_layer_call_and_return_conditional_losses_4412lç8¢5
.¢+
!
input_3ÿÿÿÿÿÿÿÿÿ
p 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 °
A__inference_model_2_layer_call_and_return_conditional_losses_6100kç7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 °
A__inference_model_2_layer_call_and_return_conditional_losses_6142kç7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
&__inference_model_2_layer_call_fn_4458_ç8¢5
.¢+
!
input_3ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
&__inference_model_2_layer_call_fn_4503_ç8¢5
.¢+
!
input_3ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
&__inference_model_2_layer_call_fn_6155^ç7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
&__inference_model_2_layer_call_fn_6168^ç7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ±
A__inference_model_3_layer_call_and_return_conditional_losses_4606lè8¢5
.¢+
!
input_4ÿÿÿÿÿÿÿÿÿ
p

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ±
A__inference_model_3_layer_call_and_return_conditional_losses_4638lè8¢5
.¢+
!
input_4ÿÿÿÿÿÿÿÿÿ
p 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 °
A__inference_model_3_layer_call_and_return_conditional_losses_6210kè7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 °
A__inference_model_3_layer_call_and_return_conditional_losses_6252kè7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
&__inference_model_3_layer_call_fn_4684_è8¢5
.¢+
!
input_4ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
&__inference_model_3_layer_call_fn_4729_è8¢5
.¢+
!
input_4ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
&__inference_model_3_layer_call_fn_6265^è7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
&__inference_model_3_layer_call_fn_6278^è7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¤
E__inference_normalize_1_layer_call_and_return_conditional_losses_6450[+¢(
!¢

xÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 |
*__inference_normalize_1_layer_call_fn_6459N+¢(
!¢

xÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
"__inference_signature_wrapper_5580ø.æçèéêJKWXQR]^cdjkqryz¢
¢ 
ª
&
bets
betsÿÿÿÿÿÿÿÿÿ

*
cards0 
cards0ÿÿÿÿÿÿÿÿÿ
*
cards1 
cards1ÿÿÿÿÿÿÿÿÿ"3ª0
.
dense_17"
dense_17ÿÿÿÿÿÿÿÿÿ