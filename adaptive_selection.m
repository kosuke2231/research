%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SMI,CCM-SMI,bind
% これまでのpropSMI,CCM削除バージョン
% 
% 
% 
% 送信はいつものシンボル×キャリア×ユーザ
% 
% 動かすには関数
% signal_generater
% tx_prepare
% singlepath
% shift
% rx_prepare
% dem_mod
% demodulation
% ber_calculaton
% tocf
% が必要(ifft,fftの転置のところがちょっと怪しい)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % 軸のフォントサイズ
% set(0, 'defaultAxesFontSize', 16);
% set(0, 'defaultAxesFontName', 'Arial');
% % set(0, 'defaultAxesFontName', 'Times');
% 
% % タイトル、注釈などのフォントサイズ
% set(0, 'defaultTextFontSize', 16);
% 
% % 軸の線の太さ
% set(0, 'DefaultLineLineWidth', 1)
% set(0, 'DefaultAxesLineWidth', 1);

clear;
% close all;
tic;
rand('state',sum(100*clock)); %rand関数の初期化

%% 値の定義
antenna_form=2;                         %1->linear, 2->rectangular
number_ant_x=8;
number_ant_y=8;
number_ant=number_ant_x*number_ant_y;                           %アンテナ数(x*yになるようにしておく)
number_pilot=16;                         %パイロット数


Doppler=0.01;              %0.01               %ドップラー周波数(10Hz)   なぜか0.001以下だと悪くなる？0.01までは普通によくなる

ebno_min=0;                             %最小Eb/No値
ebno_max=15;                            %最大Eb/No値
ebno_step=3;                            %Eb/No間隔

% number_bind=2;

repeat=100;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% この下は基本いじらない(ユーザ数kはOK)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


guard_interval=16;                       %ガードインターバル
trans_rate=2*10^6;                      %伝送帯域幅(2MHz)
number_symbol=20;                       %シンボル数
number_carrier=64;                      %キャリヤ数
ifft_size=64;                           %IFFTサイズ

D=1;                                    %Desired_User
% number_user=length(DOAvector);          %ユーザ数(=到来角数)
deg2rad=pi/180;
d=0.5;                                  %アンテナ間隔

modulation_index=4;                     %変調方式(4->16QAM)
path_index=3;                           %1->シングルパス，2->マルチパス，3->rice
number_path=5;
path_interval=1;
decay_factor=1;
K_dB=20;
K=10^(K_dB/10);
sigma=0.2;


%メモリ確保
% pilot=zeros(number_user,number_carrier,number_user);
% faded_signal=zeros(number_user,(number_carrier+guard_interval)*(number_symbol+number_pilot));
% channel=zeros(number_ant,number_user,number_carrier);
Rec_smi=zeros(number_carrier,number_symbol+number_pilot);
Rec_ccm=zeros(number_carrier,number_symbol+number_pilot);
Rec_bind=zeros(number_carrier,number_symbol+number_pilot);
Rec_zf=zeros(number_carrier,number_symbol+number_pilot);


for k=[16]
    
if k==1
    DOAvector=[40];
elseif k==2
%     DOAvector=[40,30];
    DOAvector=[40,-30];
elseif k==4
%     DOAvector=[40,30,-40,70];
    DOAvector=[40,-30,-70,80];
    DOAvector_x=[-30,40,-70,80];    %theta
    DOAvector_y=[0,-30,20,-50];    %phi
elseif k==8
%     DOAvector=[40,30,-40,70,10,-10,80,-60];
%     DOAvector=[40,-30,-70,80,60,20,-10,-50];
    DOAvector_x=[-30,40,-70,80,60,20,-10,-50];
    DOAvector_y=[0,-30,20,-50,60,10,-60,-10];
elseif k==16
    DOAvector_x=[-30,-80,-70,-60,-50,-40,-20,-10,0,10,20,30,40,50,60,70];
    DOAvector_y=[0,20,-40,50,-30,0,10,-10,-20,30,20,-30,40,-50,-20,10];
elseif k==32
    DOAvector_x=[-30,-80,-70,-60,-50,-40,-20,-10,0,10,20,30,40,50,60,70,-35,-85,-75,-65,-55,-45,-25,-15,5,15,25,35,45,55,65,75];
    DOAvector_y=[0,20,-40,50,-30,0,10,-10,-20,30,20,-30,40,-50,-20,10,5,25,-45,55,-35,5,15,-15,-25,35,25,-35,45,-55,-25,15];
end

number_user=length(DOAvector_x);

pilot=zeros(number_user,number_carrier,number_user);
%パイロット信号作成
hadamard_m=hadamard(number_user);
for nu=1:number_user
    pilot(:,:,nu)=repmat(hadamard_m(:,nu),1,ifft_size);
end

if k==1
    pilot_signal=ones(number_pilot,number_carrier,number_user);
elseif k==2
    if number_pilot==number_user
        pilot_signal=pilot;
    elseif number_pilot>number_user
        %number_pilotがuser以上のとき複製
        pilot_signal=[repmat(pilot(1,:,:),number_pilot/2,1);repmat(pilot(2,:,:),number_pilot/2,1)];
    end
elseif k==4
    if number_pilot==number_user
        pilot_signal=pilot;
    elseif number_pilot>number_user
        %number_pilotがuser以上のとき複製
        pilot_signal=[repmat(pilot(1,:,:),number_pilot/4,1);repmat(pilot(2,:,:),number_pilot/4,1);repmat(pilot(3,:,:),number_pilot/4,1);repmat(pilot(4,:,:),number_pilot/4,1)];
    end
elseif k==8
    pilot_signal=pilot;
elseif k==16
    pilot_signal=pilot;
elseif k==32
    pilot_signal=pilot;
end
    

ber_result_smi=zeros(floor((ebno_max-ebno_min)/ebno_step)+1,2);%BER測定値記録メモリ
ber_result_ccm=zeros(floor((ebno_max-ebno_min)/ebno_step)+1,2);%BER測定値記録メモリ
ber_result_bind=zeros(floor((ebno_max-ebno_min)/ebno_step)+1,2);%BER測定値記録メモリ
ber_result_zf=zeros(floor((ebno_max-ebno_min)/ebno_step)+1,2);%BER測定値記録メモリ
thpt_result_smi=zeros(floor((ebno_max-ebno_min)/ebno_step)+1,2);%Thpt測定値記録メモリ
thpt_result_ccm=zeros(floor((ebno_max-ebno_min)/ebno_step)+1,2);%Thpt測定値記録メモリ
thpt_result_bind=zeros(floor((ebno_max-ebno_min)/ebno_step)+1,2);%Thpt測定値記録メモリ
thpt_result_zf=zeros(floor((ebno_max-ebno_min)/ebno_step)+1,2);%Thpt測定値記録メモリ


for ebno=1:((ebno_max-ebno_min)/ebno_step)+1
    eb_no=(ebno-1)*ebno_step+ebno_min;
    
    modified_eb_no=eb_no+10*(log10(modulation_index)+log10(number_carrier/(number_carrier+guard_interval))+log10(number_symbol/(number_symbol+number_pilot)));%-log10(number_ant));%アンテナゲイン(変調方式に対する補正)   
%     modified_eb_no=eb_no+10*(log10(modulation_index)+log10(number_carrier/(number_carrier+guard_interval))+log10(number_symbol/(number_symbol+number_pilot))-log10(number_ant));%アンテナ正規化(変調方式に対する補正)
    
    for re=1:repeat
        %% 送信機
        disp(['Eb/N0:', num2str(eb_no) ' USER',num2str(k) ' Repeat', num2str(re)]);
        %送信信号作成
        [signal,Desired_bit_signal]=signal_generater(modulation_index,number_symbol,number_carrier,number_user,D);
            
        
        %パイロット付加した送信信号
        S=[pilot_signal;signal];
        
        %IFFT(freq2time),ガードインターバル付加,P/S変換
        serial_signal_tx=tx_prepare(S,ifft_size,guard_interval,number_user);
        
        
        %% 伝送路
        %AWGN
        noise_dis=10.^(-modified_eb_no/20);
        noise=sqrt(1/2)*(randn(number_ant,size(serial_signal_tx,2))+1j*randn(number_ant,size(serial_signal_tx,2))).*noise_dis;
        
%         Pn=10^(-modified_eb_no/10);
        % noise=(randn(number_ant,size(serial_signal_tx,2))+1j*randn(number_ant,size(serial_signal_tx,2)))*sqrt(Pn/2);      %同じ意味(教科書ver)
        
        %リニアアレイアンテナファクタ(ステアリングベクトル)
        if antenna_form==1
            PSI=exp(-1j*2*pi*d*sin(DOAvector*deg2rad));
            array_resp=(ones(number_ant,1)*PSI).^((0:(number_ant-1)).'*ones(1,number_user));
        elseif antenna_form==2
            PSI_x=exp(-1j*2*pi*d*sin(DOAvector_x*deg2rad));
            array_resp_x=(ones(number_ant_x,1)*PSI_x).^((0:(number_ant_x-1)).'*ones(1,number_user));
            PSI_y=exp(-1j*2*pi*d*sin(DOAvector_y*deg2rad));
            array_resp_y=(ones(number_ant_y,1)*PSI_y).^((0:(number_ant_y-1)).'*ones(1,number_user));
            array_resp=zeros(number_ant,number_user);
            na=1;
            for ny=1:number_ant_y
                for nx=1:number_ant_x
                    array_resp(na,:)=array_resp_x(nx,:).*array_resp_y(ny,:);
                    na=na+1;
                end
            end
        end
        
        faded_Rayleigh=zeros(number_ant,size(serial_signal_tx,2));
        if path_index==1
            %シングルパス
            for nu=1:number_user
                faded_signal(nu,:)=singlepath_LoS(serial_signal_tx(nu,:),trans_rate,Doppler);
            end
            %信号にアンテナファクタかけてAWGN付加したものが受信される(振幅をsqrt(ant)で割って正規化->電力は1/ant)
            serial_signal_rx=(array_resp*faded_signal+noise);%./sqrt(number_ant);
            
        elseif path_index==2
            %マルチパス
            for nu=1:number_user
                faded_signal(nu,:)=multipath(serial_signal_tx(nu,:),trans_rate,Doppler,number_path,decay_factor,path_interval);
            end    
            %信号にアンテナファクタかけてAWGN付加したものが受信される(振幅をsqrt(ant)で割って正規化->電力は1/ant)
            serial_signal_rx=(array_resp*faded_signal+noise);%./sqrt(number_ant);
            
        elseif path_index==3%%%%%%もしかして関数分けてやるとmultipathの1パスめとLoSのpath_amplitudeと到達タイミング同じになる？？？？
            %LoS
            for nu=1:number_user
%                 frame=71.4*10^(-6);     %LTE,5G
                frame=4*10^(-6);     %802.11
                t=frame/(guard_interval+number_carrier); %1シンボル時間?
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                 全部で遅れ
                time=[0:t:frame*(number_pilot+number_symbol)-t];
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                tx_fading=serial_signal_tx.*exp(1j*2*pi*Doppler*repmat(time,number_user,1));    %全員同じ移動（速度はともかく方向も）になっている？multipathのときは関係なかったけどいいのか…？
            end
%             faded_LoS=array_resp*serial_signal_tx;
            faded_LoS=array_resp*tx_fading;
            %multipath(LoS加えてパス数になるようにnumber_path-1にしている)
            for na=1:number_ant
                for nu=1:number_user
                    faded_Rayleigh(na,:)=faded_Rayleigh(na,:)+multipath(serial_signal_tx(nu,:),trans_rate,Doppler,number_path-1,decay_factor,path_interval);
                end
            end
            faded_signal=sqrt(K/(1+K))*faded_LoS+sqrt(1/(1+K))*faded_Rayleigh;
            serial_signal_rx=(faded_signal+noise);%./sqrt(number_ant);
        end
        
        
        
        
        %% 受信機
        %S/P変換,ガードインターバル除去,FFT(time2freq),並べ替え
        fft_sig=rx_prepare(serial_signal_rx,ifft_size,guard_interval,number_ant,number_carrier,number_symbol,number_pilot);
        
        %チャネル行列計算%
        H_resp=fft_sig(:,1:number_pilot,:);
        %pilot平均化
        Hresp=zeros(number_ant,number_user,number_carrier);
        for j=1:size(Hresp,2)
            Hresp(:,j,:)=sum(H_resp(:,number_pilot*(j-1)/number_user+1:number_pilot*j/number_user,:),2)/(number_pilot/number_user);
        end

        for nc=1:number_carrier
            ref(:,:,nc)=Hresp(:,:,nc)*inv(hadamard_m);
        end
        
        %% 検出法
        %送信信号S(シンボル×キャリア×ユーザ)->signal_tx(Desired_User,シンボル×キャリア)に変換
        signal_tx=reshape(S(:,:,D),1,number_symbol+number_pilot,number_carrier);
        %signal_txのパイロットだけ抜き出し
        pilot_tx=signal_tx(:,1:number_pilot,:);
        
        for nc=1:number_carrier
            Xt=fft_sig(:,:,nc);
            %平均化
%             Xtave=Hresp(:,:,nc);

            
           %% SMI%
            %初期解推定
            %相関行列
            %正しい
            Psmi=Xt(:,1:number_pilot)*Xt(:,1:number_pilot)'/number_pilot;
%             Psmi=Xtave*Xtave'/number_user;
            %安？
%             Psmi=Xt*Xt'/(number_symbol+number_pilot);
            %相関ベクトル
            Vsmi=Xt(:,1:number_pilot)*conj(pilot_signal(:,nc,D))/number_pilot;
%             Vsmi=Xtave*conj(pilot_signal(1:4,nc,D))/number_user;
            %重み計算
            Ws=pinv(Psmi)*Vsmi;  
%             Ws=pinv(Psmi)*ref(:,D,nc);  
            Wsmi=(Ws./(Vsmi'*Ws));
            
            
            %compensation(補償)
            Rec_smi(nc,:)=Wsmi'*Xt;
            
            
%             %% ZF(MRC)
%             %compensation
%             H=ref(:,:,nc)*ref(:,:,nc)';
%             Rec_alluser=H*Xt;
% %             Rec_alluser=pinv(ref(:,:,nc))*Xt;
%             Rec_zf(nc,:)=Rec_alluser(D,:);
            
        end
        
        %% bind SMI(交互に入れる)
        nc=1;
        while nc<=64
            for number_bind_tmp=[1,2,4,8,16,32,64]
                %         for nc=1:number_bind:number_carrier
                %number_bind本束ねたとき
                Xbind_tmp=zeros(number_ant,number_pilot*number_bind_tmp);
                cnt=1;
                for np=1:number_pilot
                    for nb=1:number_bind_tmp
                        carrier=nc+nb-1;
                        if carrier>64
                            carrier=64;
                        end
                        Xbind_tmp(:,cnt)=fft_sig(:,np,carrier);
%                         if nc==64 && carrier==64
%                             break;      %%%%要修正(nc=64のときcarrier64でcntが増えないのでXbind_tmp1列目に全部上書きされてしまう、下の二つも直す)
%                         end
                        cnt=cnt+1;
                    end
                 end
                Xbind_tmp_abs=abs(Xbind_tmp);
                Xbind_tmp_std=zeros(number_ant,number_pilot);
                for np=1:number_pilot
                    Xbind_tmp_std(:,np)=std(Xbind_tmp_abs(:,(np-1)*number_bind_tmp+1:np*number_bind_tmp),0,2);
                end
%                 Xbind_tmp_std_mean=mean(Xbind_tmp_std,'all');%%%平均でいいかは吟味の余地あり（誤差だから最大値を評価基準にするとか）
                Xbind_tmp_std_mean=mean(mean(Xbind_tmp_std),2);
                if Xbind_tmp_std_mean > sigma
                    break;
                elseif nc+number_bind_tmp*2-1 > 64%nc==64 && carrier==64    %サブキャリア数超えそうだったら(最終サブキャリアまでいってしまったら)bind終わり
                    number_bind_tmp=number_bind_tmp*2;
                    break;
                end
            end
            %%%%%%%%    breakした後使いたいXbindは上書きされちゃってるからもう一回計算する、あとはこれをncループの中にうまく入れ込む
            
            number_bind=number_bind_tmp/2;      %これだと64にはならないけど、実用上そんな大きな問題ではないからいいか。
            Xbind=zeros(number_ant,number_pilot*number_bind);
            cnt=1;
            for np=1:number_pilot
                for nb=1:number_bind
                    carrier=nc+nb-1;
                    if carrier>64
                        carrier=64;
                    end
                    Xbind(:,cnt)=fft_sig(:,np,carrier);
                    cnt=cnt+1;
                end
            end
            
            
            %bind平均化   %←bind数平均化も抜いてみたけどあまりよくない？？(313~317)<-正方行列になってしまうときがあるから？
            Xbind_ave=zeros(number_ant,number_pilot);
            for i=1:size(Xbind_ave,2)
                Xbind_ave(:,i)=sum(Xbind(:,(i-1)*number_bind+1:i*number_bind),2)/number_bind;
            end
%             %             %pilot平均化
%             %             Xbindave=zeros(number_ant,number_user);
%             %             for j=1:size(Xbindave,2)
%             %                 Xbindave(:,j)=sum(Xbind_ave(:,number_pilot*(j-1)/number_user+1:number_pilot*j/number_user),2)/(number_pilot/number_user);
%             %             end
%             
%             %             Pbind=Xbindave*Xbindave'/number_user;
            Pbind=Xbind_ave*Xbind_ave'/number_pilot;
%             Pbind=Xbind*Xbind'/(number_pilot*number_bind);
            Vbind=Xbind*conj(repmat(pilot_signal(:,nc,D),number_bind,1))/(number_pilot*number_bind);        %D=1じゃないときは書き換え必要な気がする(repmat)
            Wb=pinv(Pbind)*Vbind;
            Wbind=(Wb./(Vbind'*Wb));
            
            %compensation(補償)
            for nb=1:number_bind
                carrier=nc+nb-1;
                if carrier>64
                    carrier=64;
                end
                Rec_bind(nc+nb-1,:)=Wbind'*fft_sig(:,:,carrier);
            end
            
            nc=nc+number_bind;
        end
        
        
        %% CCM-SMI%
        %相関行列
        %正しい
        Pccm=serial_signal_rx(:,1:((guard_interval+number_carrier)*number_pilot))*serial_signal_rx(:,1:((guard_interval+number_carrier)*number_pilot))'/((guard_interval+number_carrier)*number_pilot);
        %安？
%         Pccm=serial_signal_rx*serial_signal_rx'/((guard_interval+number_carrier)*(number_symbol+number_pilot));
        %相関ベクトル
        Vccm=serial_signal_rx(:,1:((guard_interval+number_carrier)*number_pilot))*conj(serial_signal_tx(D,1:(guard_interval+number_carrier)*number_pilot).')/((guard_interval+number_carrier)*number_pilot);
        %重み
        Wc=pinv(Pccm)*Vccm;
        Wccm=(Wc./(Vccm'*Wc));
        
        %compensation(補償)
        for nc=1:number_carrier
            Rec_ccm(nc,:)=Wccm'*fft_sig(:,:,nc);
        end

        
        %% 復調,誤り計算
%         
%         W(:,1)=Wsmi(:,1);
%         W(:,3)=Wccm(:,1);
        
        %Rec(carrier*symbol)->Recover(symbol*carrier)へ
        Recover_smi=Rec_smi.';
        Recover_ccm=Rec_ccm.';
        Recover_bind=Rec_bind.';
%         Recover_zf=Rec_zf.';    %%%%
        
        detected_bit_smi=demodulation(modulation_index,Recover_smi, number_symbol, number_pilot, number_carrier);
        detected_bit_ccm=demodulation(modulation_index,Recover_ccm, number_symbol, number_pilot, number_carrier);
        detected_bit_bind=demodulation(modulation_index,Recover_bind, number_symbol, number_pilot, number_carrier);
%          detected_bit_zf=demodulation(modulation_index,Recover_zf, number_symbol, number_pilot, number_carrier);    %%%%
        
        %BER計算
        ber_result_smi=ber_calculation(modulation_index,ber_result_smi, Desired_bit_signal, detected_bit_smi, ebno, modified_eb_no);
        ber_result_ccm=ber_calculation(modulation_index,ber_result_ccm, Desired_bit_signal, detected_bit_ccm, ebno, modified_eb_no);
        ber_result_bind=ber_calculation(modulation_index,ber_result_bind, Desired_bit_signal, detected_bit_bind, ebno, modified_eb_no);
%         ber_result_zf=ber_calculation(modulation_index,ber_result_zf, Desired_bit_signal, detected_bit_zf, ebno, modified_eb_no);   %%%%

        %Thpt計算
        thpt_result_smi=thpt_calculation(modulation_index, thpt_result_smi, Desired_bit_signal, detected_bit_smi, ebno, modified_eb_no, trans_rate, serial_signal_tx);
        thpt_result_ccm=thpt_calculation(modulation_index, thpt_result_ccm, Desired_bit_signal, detected_bit_ccm, ebno, modified_eb_no, trans_rate, serial_signal_tx);
        thpt_result_bind=thpt_calculation(modulation_index, thpt_result_bind, Desired_bit_signal, detected_bit_bind, ebno, modified_eb_no, trans_rate, serial_signal_tx);
%         thpt_result_zf=thpt_calculation(modulation_index, thpt_result_zf, Desired_bit_signal, detected_bit_proccm, ebno, modified_eb_no, trans_rate, serial_signal_tx);
    end
end
%% プロット
ber_result_smi(:,2)=ber_result_smi(:,2)./repeat;
ber_result_ccm(:,2)=ber_result_ccm(:,2)./repeat;
ber_result_bind(:,2)=ber_result_bind(:,2)./repeat;
% ber_result_zf(:,2)=ber_result_zf(:,2)./repeat;  %%%%

figure(1);
semilogy(ebno_min:ebno_step:ebno_max,ber_result_smi(:,2)','ro--');
hold on;
semilogy(ebno_min:ebno_step:ebno_max,ber_result_ccm(:,2)','b<-');
semilogy(ebno_min:ebno_step:ebno_max,ber_result_bind(:,2)','g^-');
% semilogy(ebno_min:ebno_step:ebno_max,ber_result_zf(:,2)','kx-');    %%%%

legend('MMSE-SMI' , 'CCM-SMI','bind');
% legend('MMSE-SMI' , 'CCM-SMI','bind','MRC');

title('BER performance');
xlabel('Eb/No [dB]');
ylabel('BER');
axis([0,15,10^(-5),10^(0)]);
grid on;

%Thptプロット
thpt_result_smi(:,2)=thpt_result_smi(:,2)./repeat;
thpt_result_ccm(:,2)=thpt_result_ccm(:,2)./repeat;
thpt_result_bind(:,2)=thpt_result_bind(:,2)./repeat;
% thpt_result_zf(:,2)=thpt_result_zf(:,2)./repeat; 

figure(2);
plot(ebno_min:ebno_step:ebno_max,thpt_result_smi(:,2)','ro-');
hold on;
plot(ebno_min:ebno_step:ebno_max,thpt_result_ccm(:,2)','b^-');
plot(ebno_min:ebno_step:ebno_max,thpt_result_bind(:,2)','g<-');

legend('MMSE-SMI' , 'CCM-SMI','bind');
title('Throughput');
xlabel('Eb/No [dB]');
ylabel('Throughput');
axis([0,15,0,10^7]);
grid on;


%BER保存
% saveas(gcf,num2str(K_dB));
% saveas(gcf,'K50fd001sigma02');


end



% %アンテナパターン%(30dBのとき.repeat分平均化.SMIはcarrier64のとき)
% WW=W./repeat;
% doa=-90:90;     %Degree of Arrival
% figure(2);
% ant_pattern(doa,d,number_ant,WW(:,4));      %proccmのアンテナパターン
% title('Prop.CCM-SMI');
% figure(3);
% ant_pattern(doa,d,number_ant,WW(:,3));      %ccmのアンテナパターン
% title('Conv.CCM-SMI');
% figure(4);
% ant_pattern(doa,d,number_ant,WW(:,2));      %prosmiのアンテナパターン
% title('Prop.SMI');
% figure(5);
% ant_pattern(doa,d,number_ant,WW(:,1));      %smiのアンテナパターン
% title('Conv.SMI');
% 
% figure(6);
% plot(Recover_ccm(number_pilot+1:number_pilot+number_symbol,:),'.');
% title('Conv.CCM-SMI');
% axis([-1.5,1.5,-1.5,1.5]);
% grid on;

% figure(9);
% plot(Recover_smi(number_pilot+1:number_pilot+number_symbol,:),'.');
% title('MMSE-SMI');
% axis([-1.5,1.5,-1.5,1.5]);
% grid on;
% 
% 
% figure(10);
% plot(Recover_bind(number_pilot+1:number_pilot+number_symbol,:),'.');
% title('bind');
% axis([-1.5,1.5,-1.5,1.5]);
% grid on;


tocf(3);